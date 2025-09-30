#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RV32I Two-Pass Assembler — v2 (con %hi/%lo por regex)
=====================================================
- Reconoce TODAS las instrucciones base RV32I por tipo: R, I, S, B, U, J.
- Soporta loads/stores, saltos y ramas, aritmética y lógicas, jal/jalr, lui/auipc.
- Soporta pseudoinstrucciones comunes (nop, li, mv, j, jr, ret, beqz, bnez, not, neg, seqz, snez, sltz, sgtz).
- Dos pasadas: 1) construye tabla de símbolos; 2) codifica (resuelve labels).
- Directivas: .text (0x00000000) y .data (registra labels, base simbólica 0x10000000).
- Salida: un word de 32 bits por línea en HEX y BIN (agrupado por campos).
- NUEVO: Soporte de relocaciones %hi(label) y %lo(label) mediante expresiones regulares.

Uso:
  python assembler.py program.asm program.hex program.bin
"""

from __future__ import annotations
import sys, re
from typing import List, Tuple, Dict

# -----------------------------------------------------------------------------
# Errores
# -----------------------------------------------------------------------------
class AsmError(Exception):
    pass

# -----------------------------------------------------------------------------
# Utilidades y registros 
# -----------------------------------------------------------------------------
REGISTROS_ALIAS: Dict[str,int] = {
    "zero":0, "ra":1, "sp":2, "gp":3, "tp":4,
    "t0":5, "t1":6, "t2":7,
    "s0":8, "fp":8, "s1":9,
    "a0":10, "a1":11, "a2":12, "a3":13, "a4":14, "a5":15, "a6":16, "a7":17,
    "s2":18, "s3":19, "s4":20, "s5":21, "s6":22, "s7":23, "s8":24, "s9":25, "s10":26, "s11":27,
    "t3":28, "t4":29, "t5":30, "t6":31
}

def parse_reg(tok: str) -> int:
    t = tok.strip().lower()
    if t.startswith('x') and t[1:].isdigit():
        n = int(t[1:])
        if 0 <= n <= 31: return n
    if t in REGISTROS_ALIAS: return REGISTROS_ALIAS[t]
    raise AsmError(f"Registro inválido: {tok}")

def parse_imm(s: str) -> int:
    t = s.strip().lower().replace('_','')
    neg = False
    if t.startswith('-'):
        neg = True; t = t[1:]
    base = 10
    if t.startswith('0x'):
        base = 16; t = t[2:]
    elif t.startswith('0b'):
        base = 2; t = t[2:]
    elif t.startswith('0o'):
        base = 8; t = t[2:]
    try:
        val = int(t, base)
    except ValueError:
        raise AsmError(f"Inmediato inválido: {s}")
    return -val if neg else val

def imm_in_range(val: int, bits: int, signed: bool=True) -> bool:
    if signed:
        lo = -(1 << (bits-1))
        hi = (1 << (bits-1)) - 1
        return lo <= val <= hi
    else:
        return 0 <= val <= (1<<bits)-1

def sext(val: int, bits: int) -> int:
    mask = (1<<bits) - 1
    val &= mask
    if val & (1<<(bits-1)):
        val -= (1<<bits)
    return val

def to_hex8(word: int) -> str:
    return format(word & 0xFFFFFFFF, '08x')

def to_bin32(word: int) -> str:
    return format(word & 0xFFFFFFFF, '032b')

# ===== BIN agrupado por formato (R/I/S/B/U/J) =====
def format_bin_grouped(word: int) -> str:
    """Devuelve los 32 bits agrupados por campos según el formato RV32I."""
    def bits(val, hi, lo):
        return (val >> lo) & ((1 << (hi - lo + 1)) - 1)

    opcode = bits(word, 6, 0)

    # R-type
    if opcode == OP_R:
        f7   = bits(word, 31, 25)
        rs2  = bits(word, 24, 20)
        rs1  = bits(word, 19, 15)
        f3   = bits(word, 14, 12)
        rd   = bits(word, 11, 7)
        return f"{f7:07b} {rs2:05b} {rs1:05b} {f3:03b} {rd:05b} {opcode:07b}"

    # I-type (arith/logic/loads/jalr/system/fence)
    if opcode in (OP_I, OP_L, OP_I_JALR, OP_SYS, OP_FENCE):
        imm  = bits(word, 31, 20)
        rs1  = bits(word, 19, 15)
        f3   = bits(word, 14, 12)
        rd   = bits(word, 11, 7)
        return f"{imm:012b} {rs1:05b} {f3:03b} {rd:05b} {opcode:07b}"

    # S-type
    if opcode == OP_S:
        imm11_5 = bits(word, 31, 25)
        rs2     = bits(word, 24, 20)
        rs1     = bits(word, 19, 15)
        f3      = bits(word, 14, 12)
        imm4_0  = bits(word, 11, 7)
        return f"{imm11_5:07b} {rs2:05b} {rs1:05b} {f3:03b} {imm4_0:05b} {opcode:07b}"

    # B-type
    if opcode == OP_B:
        b12   = bits(word, 31, 31)
        b10_5 = bits(word, 30, 25)
        rs2   = bits(word, 24, 20)
        rs1   = bits(word, 19, 15)
        f3    = bits(word, 14, 12)
        b4_1  = bits(word, 11, 8)
        b11   = bits(word, 7, 7)
        return f"{b12:01b} {b10_5:06b} {rs2:05b} {rs1:05b} {f3:03b} {b4_1:04b} {b11:01b} {opcode:07b}"

    # U-type (lui/auipc)
    if opcode in (OP_U_LUI, OP_U_AUIPC):
        imm31_12 = bits(word, 31, 12)
        rd       = bits(word, 11, 7)
        return f"{imm31_12:020b} {rd:05b} {opcode:07b}"

    # J-type
    if opcode == OP_J:
        j20    = bits(word, 31, 31)
        j10_1  = bits(word, 30, 21)
        j11    = bits(word, 20, 20)
        j19_12 = bits(word, 19, 12)
        rd     = bits(word, 11, 7)
        return f"{j20:01b} {j10_1:010b} {j11:01b} {j19_12:08b} {rd:05b} {opcode:07b}"

    return to_bin32(word)
# ===================================================

# -----------------------------------------------------------------------------
# Encoders por formato
# -----------------------------------------------------------------------------
def enc_R(funct7, rs2, rs1, funct3, rd, opcode):
    return ((funct7 & 0x7f) << 25) | ((rs2 & 0x1f) << 20) | ((rs1 & 0x1f) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1f) << 7) | (opcode & 0x7f)

def enc_I(imm, rs1, funct3, rd, opcode):
    imm &= 0xFFF
    return (imm << 20) | ((rs1 & 0x1f) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1f) << 7) | (opcode & 0x7f)

def enc_S(imm, rs2, rs1, funct3, opcode):
    imm &= 0xFFF
    imm11_5 = (imm >> 5) & 0x7F
    imm4_0  = imm & 0x1F
    return (imm11_5 << 25) | ((rs2 & 0x1f) << 20) | ((rs1 & 0x1f) << 15) | ((funct3 & 0x7) << 12) | (imm4_0 << 7) | (opcode & 0x7f)

def enc_B(imm, rs2, rs1, funct3, opcode):
    imm = sext(imm, 13)
    b12   = (imm >> 12) & 1
    b10_5 = (imm >> 5) & 0x3f
    b4_1  = (imm >> 1) & 0xf
    b11   = (imm >> 11) & 1
    return (b12 << 31) | (b10_5 << 25) | ((rs2 & 0x1f) << 20) | ((rs1 & 0x1f) << 15) | ((funct3 & 0x7) << 12) | (b4_1 << 8) | (b11 << 7) | (opcode & 0x7f)

def enc_U(imm, rd, opcode):
    return (imm & 0xFFFFF000) | ((rd & 0x1f) << 7) | (opcode & 0x7f)

def enc_J(imm, rd, opcode):
    imm = sext(imm, 21)
    j20    = (imm >> 20) & 1
    j10_1  = (imm >> 1) & 0x3ff
    j11    = (imm >> 11) & 1
    j19_12 = (imm >> 12) & 0xff
    return (j20 << 31) | (j19_12 << 12) | (j11 << 20) | (j10_1 << 21) | ((rd & 0x1f) << 7) | (opcode & 0x7f)

# -----------------------------------------------------------------------------
# Tablas ISA — RV32I base
# -----------------------------------------------------------------------------
OP_R      = 0b0110011
OP_I      = 0b0010011
OP_L      = 0b0000011
OP_S      = 0b0100011
OP_B      = 0b1100011
OP_J      = 0b1101111
OP_I_JALR = 0b1100111
OP_U_LUI  = 0b0110111
OP_U_AUIPC= 0b0010111
OP_SYS    = 0b1110011
OP_FENCE  = 0b0001111

R_FUNCTS = {
    "add":  (0b0000000, 0b000),
    "sub":  (0b0100000, 0b000),
    "sll":  (0b0000000, 0b001),
    "slt":  (0b0000000, 0b010),
    "sltu": (0b0000000, 0b011),
    "xor":  (0b0000000, 0b100),
    "srl":  (0b0000000, 0b101),
    "sra":  (0b0100000, 0b101),
    "or":   (0b0000000, 0b110),
    "and":  (0b0000000, 0b111),
}

I_FUNCTS = {
    "addi": 0b000, "slti": 0b010, "sltiu":0b011,
    "xori": 0b100, "ori":  0b110, "andi": 0b111,
    "slli": 0b001, "srli": 0b101, "srai": 0b101,
}

L_FUNCTS = {
    "lb":  0b000, "lh":  0b001, "lw":  0b010,
    "lbu": 0b100, "lhu": 0b101,
}

S_FUNCTS = { "sb":0b000, "sh":0b001, "sw":0b010 }

B_FUNCTS = {
    "beq":0b000, "bne":0b001, "blt":0b100,
    "bge":0b101, "bltu":0b110, "bgeu":0b111,
}

# -----------------------------------------------------------------------------
# Parsing de líneas (expresiones regulares)
# -----------------------------------------------------------------------------
COMMENT_RE = re.compile(r'(#|//).*')
LABEL_RE   = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*):')
WS_COMMA   = re.compile(r'\s*,\s*')

# Relocaciones %hi(label) y %lo(label)
RE_HI = re.compile(r'^\s*%hi\(([A-Za-z_][A-Za-z0-9_]*)\)\s*$')
RE_LO = re.compile(r'^\s*%lo\(([A-Za-z_][A-Za-z0-9_]*)\)\s*$')

def eval_reloc(expr: str, symbols: Dict[str,int]) -> int:
    """
    Evalúa %hi(label) y %lo(label) usando la tabla de símbolos.
    - %hi(label) = (addr(label) + 0x800) >> 12
    - %lo(label) = addr(label) & 0xFFF  (el signo lo maneja la I-type)
    Lanza ValueError si 'expr' no es una reloc %hi/%lo.
    """
    m = RE_HI.match(expr)
    if m:
        lab = m.group(1)
        if lab not in symbols:
            raise AsmError(f"Etiqueta no definida en %hi(): '{lab}'")
        addr = symbols[lab]
        return (addr + 0x800) >> 12

    m = RE_LO.match(expr)
    if m:
        lab = m.group(1)
        if lab not in symbols:
            raise AsmError(f"Etiqueta no definida en %lo(): '{lab}'")
        addr = symbols[lab]
        return addr & 0xFFF

    raise ValueError("no relocation")

def strip_comment(line: str) -> str:
    m = COMMENT_RE.search(line)
    if m: line = line[:m.start()]
    return line.strip()

def split_operands(s: str) -> List[str]:
    return [] if not s else WS_COMMA.split(s)

# -----------------------------------------------------------------------------
# Ensamblador de 2 pasadas
# -----------------------------------------------------------------------------
class Assembler:
    def __init__(self, lines: List[str]):
        self.lines   = lines
        self.symbols: Dict[str,int] = {}
        self.segment = '.text'
        self.addr_text = 0x00000000
        self.addr_data = 0x10000000  # base simbólica para .data
        self.pc = self.addr_text
        self.insns: List[Tuple[int,str,int]] = []  # (addr, text, line_no)

    def pass1(self):
        self.pc = self.addr_text
        self.segment = '.text'
        for idx, raw in enumerate(self.lines, 1):
            line = strip_comment(raw)
            if not line: continue
            # Etiquetas (pueden venir varias seguidas)
            while True:
                m = LABEL_RE.match(line)
                if not m: break
                label = m.group(1)
                if label in self.symbols:
                    raise AsmError(f"Línea {idx}: etiqueta duplicada '{label}'")
                self.symbols[label] = self.pc if self.segment=='.text' else self.addr_data
                line = line[m.end():].strip()
                if not line: break
            if not line: continue
            # Directivas
            if line.startswith('.'):
                d = line.split()[0].lower()
                if d == '.text': self.segment = '.text'
                elif d == '.data': self.segment = '.data'
                else: pass
                continue
            # Instrucciones solo en .text
            if self.segment == '.text':
                self.insns.append((self.pc, line, idx))
                self.pc += 4

    def pass2(self) -> List[int]:
        words: List[int] = []
        i = 0
        while i < len(self.insns):
            addr, line, idx = self.insns[i]
            w = self.encode_line(addr, line, idx, i)
            words.append(w)
            i += 1
        return words

    # ---------------- Pseudoinstrucciones ----------------
    def expand_pseudo(self, addr: int, mnem: str, ops: List[str], idx: int) -> List[str]:
        m = mnem.lower()
        if m == 'nop':  return ['addi x0, x0, 0']
        if m == 'mv':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: mv espera 2 operandos")
            return [f"addi {ops[0]}, {ops[1]}, 0"]
        if m == 'not':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: not espera 2 operandos")
            return [f"xori {ops[0]}, {ops[1]}, -1"]
        if m == 'neg':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: neg espera 2 operandos")
            return [f"sub {ops[0]}, x0, {ops[1]}"]
        if m == 'seqz':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: seqz espera 2 operandos")
            return [f"sltiu {ops[0]}, {ops[1]}, 1"]
        if m == 'snez':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: snez espera 2 operandos")
            return [f"sltu {ops[0]}, x0, {ops[1]}"]
        if m == 'sltz':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: sltz espera 2 operandos")
            return [f"slt {ops[0]}, {ops[1]}, x0"]
        if m == 'sgtz':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: sgtz espera 2 operandos")
            return [f"slt {ops[0]}, x0, {ops[1]}"]
        if m == 'j':
            if len(ops)!=1: raise AsmError(f"Línea {idx}: j espera etiqueta")
            return [f"jal x0, {ops[0]}"]
        if m == 'jr':
            if len(ops)!=1: raise AsmError(f"Línea {idx}: jr espera registro")
            return [f"jalr x0, {ops[0]}, 0"]
        if m == 'ret':
            if len(ops)!=0: raise AsmError(f"Línea {idx}: ret no lleva operandos")
            return ["jalr x0, ra, 0"]
        if m in ('beqz','bnez'):
            if len(ops)!=2: raise AsmError(f"Línea {idx}: {m} espera 2 operandos (rs, label)")
            cond = 'beq' if m=='beqz' else 'bne'
            return [f"{cond} {ops[0]}, x0, {ops[1]}"]
        if m == 'li':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: li espera 2 operandos")
            rd = ops[0]; imm = parse_imm(ops[1])
            if imm_in_range(imm, 12, signed=True):
                return [f"addi {rd}, x0, {imm}"]
            upper = (imm + (1<<11)) >> 12
            lower = imm - (upper << 12)
            upper &= 0xFFFFF
            return [f"lui {rd}, {upper}", f"addi {rd}, {rd}, {lower}"]
        return [mnem + (' ' + ', '.join(ops) if ops else '')]

    # ---------------- Codificación ----------------
    def encode_line(self, addr: int, line: str, idx: int, insn_index: int) -> int:
        parts = line.split(None, 1)
        if not parts:
            raise AsmError(f"Línea {idx}: vacía inesperadamente")
        mnem = parts[0].lower()
        ops  = split_operands(parts[1]) if len(parts)>1 else []

        # Expansión de pseudo
        expanded = self.expand_pseudo(addr, mnem, ops, idx)
        if len(expanded) > 1:
            first, rest = expanded[0], expanded[1:]
            cur_addr = addr + 4
            insert_at = insn_index + 1
            for text in rest:
                self.insns.insert(insert_at, (cur_addr, text, idx))
                insert_at += 1; cur_addr += 4
            for j in range(insert_at, len(self.insns)):
                a,l,i2 = self.insns[j]
                self.insns[j] = (a + 4*len(rest), l, i2)
            for k,v in list(self.symbols.items()):
                if v > addr:
                    self.symbols[k] = v + 4*len(rest)
            return self.encode_line(addr, first, idx, insn_index)

        parts = expanded[0].split(None, 1)
        mnem = parts[0].lower()
        ops  = split_operands(parts[1]) if len(parts)>1 else []

        if mnem.startswith('.'):
            return 0

        # ----- R-type -----
        if mnem in R_FUNCTS:
            if len(ops)!=3: raise AsmError(f"Línea {idx}: {mnem} espera 3 operandos (rd, rs1, rs2)")
            rd, rs1, rs2 = parse_reg(ops[0]), parse_reg(ops[1]), parse_reg(ops[2])
            funct7, funct3 = R_FUNCTS[mnem]
            return enc_R(funct7, rs2, rs1, funct3, rd, OP_R)

        # ----- I-type aritm/log (incluye shifts) -----
        if mnem in I_FUNCTS:
            f3 = I_FUNCTS[mnem]
            if mnem in ('slli','srli','srai'):
                if len(ops)!=3: raise AsmError(f"Línea {idx}: {mnem} espera 3 operandos (rd, rs1, shamt)")
                rd, rs1, sh = parse_reg(ops[0]), parse_reg(ops[1]), parse_imm(ops[2])
                if not imm_in_range(sh, 5, signed=False):
                    raise AsmError(f"Línea {idx}: desplazamiento fuera de rango (0..31)")
                funct7 = 0b0100000 if mnem=='srai' else 0b0000000
                imm = (funct7<<5) | (sh & 0x1f)
                return enc_I(imm, rs1, f3, rd, OP_I)
            else:
                if len(ops)!=3: raise AsmError(f"Línea {idx}: {mnem} espera 3 operandos (rd, rs1, imm)")
                rd, rs1 = parse_reg(ops[0]), parse_reg(ops[1])
                # ---- NUEVO: reloc %lo(label) soportado por regex ----
                try:
                    imm = eval_reloc(ops[2], self.symbols)   # intenta %lo(label)
                except ValueError:
                    imm = parse_imm(ops[2])                  # numérico normal
                if not imm_in_range(imm, 12, signed=True):
                    raise AsmError(f"Línea {idx}: inmediato fuera de rango para {mnem} (12 bits signed)")
                return enc_I(imm, rs1, f3, rd, OP_I)

        # ----- Loads (I-type) -----
        if mnem in L_FUNCTS:
            if len(ops)!=2: raise AsmError(f"Línea {idx}: {mnem} espera 2 operandos (rd, offset(rs1))")
            rd = parse_reg(ops[0])
            m = re.match(r'^(-?0x[0-9a-fA-F]+|-?\d+)\(([^)]+)\)$', ops[1].replace(' ',''))
            if not m: raise AsmError(f"Línea {idx}: formato de dirección inválido, use offset(rs1)")
            imm = parse_imm(m.group(1))
            rs1 = parse_reg(m.group(2))
            if not imm_in_range(imm, 12, signed=True):
                raise AsmError(f"Línea {idx}: inmediato fuera de rango (12 bits)")
            return enc_I(imm, rs1, L_FUNCTS[mnem], rd, OP_L)

        # ----- Stores (S-type) -----
        if mnem in S_FUNCTS:
            if len(ops)!=2: raise AsmError(f"Línea {idx}: {mnem} espera 2 operandos (rs2, offset(rs1))")
            rs2 = parse_reg(ops[0])
            m = re.match(r'^(-?0x[0-9a-fA-F]+|-?\d+)\(([^)]+)\)$', ops[1].replace(' ',''))
            if not m: raise AsmError(f"Línea {idx}: formato de dirección inválido, use offset(rs1)")
            imm = parse_imm(m.group(1))
            rs1 = parse_reg(m.group(2))
            if not imm_in_range(imm, 12, signed=True):
                raise AsmError(f"Línea {idx}: inmediato fuera de rango (12 bits)")
            return enc_S(imm, rs2, rs1, S_FUNCTS[mnem], OP_S)

        # ----- Branches (B-type) -----
        if mnem in B_FUNCTS:
            if len(ops)!=3: raise AsmError(f"Línea {idx}: {mnem} espera 3 operandos (rs1, rs2, label)")
            rs1, rs2 = parse_reg(ops[0]), parse_reg(ops[1])
            label = ops[2]
            if label not in self.symbols:
                raise AsmError(f"Línea {idx}: etiqueta no definida '{label}'")
            target = self.symbols[label]
            offset = target - addr
            if not imm_in_range(offset, 13, signed=True):
                raise AsmError(f"Línea {idx}: desplazamiento de branch fuera de rango")
            return enc_B(offset, rs2, rs1, B_FUNCTS[mnem], OP_B)

        # ----- Jumps (J-type y JALR) -----
        if mnem == 'jal':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: jal espera 2 operandos (rd, label)")
            rd = parse_reg(ops[0]); label = ops[1]
            if label not in self.symbols: raise AsmError(f"Línea {idx}: etiqueta no definida '{label}'")
            target = self.symbols[label]
            offset = target - addr
            if not imm_in_range(offset, 21, signed=True):
                raise AsmError(f"Línea {idx}: desplazamiento de JAL fuera de rango")
            return enc_J(offset, rd, OP_J)

        if mnem == 'jalr':
            if len(ops)!=3: raise AsmError(f"Línea {idx}: jalr espera 3 operandos (rd, rs1, imm)")
            rd, rs1, imm = parse_reg(ops[0]), parse_reg(ops[1]), parse_imm(ops[2])
            if not imm_in_range(imm, 12, signed=True):
                raise AsmError(f"Línea {idx}: inmediato fuera de rango (12 bits)")
            return enc_I(imm, rs1, 0b000, rd, OP_I_JALR)

        # ----- U-type (LUI/AUIPC) -----
        if mnem == 'lui' or mnem == 'auipc':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: {mnem} espera 2 operandos (rd, imm20)")
            rd = parse_reg(ops[0])
            # ---- NUEVO: reloc %hi(label) soportado por regex ----
            try:
                imm20 = eval_reloc(ops[1], self.symbols)     # %hi(label)
            except ValueError:
                imm20 = parse_imm(ops[1])                    # numérico
            if not imm_in_range(imm20, 20, signed=False):
                raise AsmError(f"Línea {idx}: inmediato de {mnem} debe caber en 20 bits sin signo")
            opcode = OP_U_LUI if mnem=='lui' else OP_U_AUIPC
            return enc_U(imm20<<12, rd, opcode)

        # ----- SYSTEM mínimos -----
        if mnem == 'ecall':  return enc_I(0, 0, 0, 0, OP_SYS)
        if mnem == 'ebreak': return enc_I(1, 0, 0, 0, OP_SYS)

        raise AsmError(f"Línea {idx}: instrucción inválida o no soportada: '{mnem}'")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv=None):
    if argv is None: argv = sys.argv
    if len(argv) != 4:
        print("Uso: python assembler.py program.asm program.hex program.bin")
        return 2
    in_path, hex_path, bin_path = argv[1], argv[2], argv[3]

    try:
        with open(in_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        print(f"Archivo no encontrado: {in_path}")
        return 1

    asm = Assembler(lines)
    try:
        asm.pass1()
        words = asm.pass2()
    except AsmError as e:
        print(f"Error de ensamblado: {e}")
        return 1

    with open(hex_path, 'w', encoding='utf-8') as fhex, open(bin_path, 'w', encoding='utf-8') as fbin:
        for w in words:
            fhex.write(to_hex8(w) + '\n')
            fbin.write(format_bin_grouped(w) + '\n')

    print(f"Ensambla OK: {len(words)} instrucciones -> {hex_path}, {bin_path}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
