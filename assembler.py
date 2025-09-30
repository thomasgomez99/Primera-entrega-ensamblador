
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RV32I Two-Pass Assembler (educational, Python 3)
------------------------------------------------
- Lee un .asm con código RV32I y produce .hex (32b/linea) y .bin (32b/linea).
- Dos pasadas: (1) construye tabla de símbolos (PC avanza de 4 en 4), (2) codifica.
- Maneja directivas básicas: .text (código desde 0x00000000), .data (etiquetas de datos).
- Expande pseudoinstrucciones comunes (nop, li, mv, j, jr, ret, beqz, bnez, not, neg, seqz, snez, sltz, sgtz).
- Reporta errores claros: sintaxis, instrucción inválida, operandos, etiqueta indefinida, rangos de inmediatos.

Uso:
    python assembler.py program.asm program.hex program.bin
"""

import sys
import re
from typing import List, Tuple, Dict

# -------------------------
# Errores
# -------------------------
class AsmError(Exception):
    pass

# -------------------------
# Registros y utilidades
# -------------------------
REG_ALIASES = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7,
    "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15, "a6": 16, "a7": 17,
    "s2": 18, "s3": 19, "s4": 20, "s5": 21, "s6": 22, "s7": 23, "s8": 24, "s9": 25, "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31,
}

def parse_reg(tok: str) -> int:
    tok = tok.strip().lower()
    if tok.startswith("x") and tok[1:].isdigit():
        n = int(tok[1:])
        if 0 <= n <= 31: return n
    if tok in REG_ALIASES: return REG_ALIASES[tok]
    raise AsmError(f"Registro inválido: {tok}")

def parse_imm(s: str) -> int:
    s = s.strip().lower().replace('_','')
    neg = False
    if s.startswith('-'):
        neg=True; s=s[1:]
    base = 10
    if s.startswith('0x'):
        base=16; s=s[2:]
    elif s.startswith('0b'):
        base=2; s=s[2:]
    elif s.startswith('0o'):
        base=8; s=s[2:]
    try:
        val = int(s, base)
    except ValueError:
        raise AsmError(f"Inmediato inválido: {s}")
    return -val if neg else val

def imm_in_range(val: int, bits: int, signed: bool=True) -> bool:
    if signed:
        minv = -(1 << (bits-1))
        maxv = (1 << (bits-1)) - 1
        return minv <= val <= maxv
    else:
        return 0 <= val <= (1<<bits)-1

def sext(val: int, bits: int) -> int:
    mask = (1<<bits)-1
    val &= mask
    if val & (1<<(bits-1)):
        val -= (1<<bits)
    return val

def to_bin32(word: int) -> str:
    return format(word & 0xFFFFFFFF, '032b')

def to_hex8(word: int) -> str:
    return format(word & 0xFFFFFFFF, '08x')

# -------------------------
# Encoders (R/I/S/B/U/J)
# -------------------------
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
    return ((imm & 0xFFFFF000)) | ((rd & 0x1f) << 7) | (opcode & 0x7f)

def enc_J(imm, rd, opcode):
    imm = sext(imm, 21)
    j20    = (imm >> 20) & 1
    j10_1  = (imm >> 1) & 0x3ff
    j11    = (imm >> 11) & 1
    j19_12 = (imm >> 12) & 0xff
    return (j20 << 31) | (j19_12 << 12) | (j11 << 20) | (j10_1 << 21) | ((rd & 0x1f) << 7) | (opcode & 0x7f)

# -------------------------
# OPCODES/FUNCTS (RV32I base)
# -------------------------
OP_R = 0b0110011
OP_I = 0b0010011
OP_L = 0b0000011
OP_S = 0b0100011
OP_B = 0b1100011
OP_J = 0b1101111
OP_I_JALR = 0b1100111
OP_U_LUI = 0b0110111
OP_U_AUIPC = 0b0010111
OP_SYS = 0b1110011
OP_FENCE = 0b0001111

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
    "addi": 0b000,
    "slti": 0b010,
    "sltiu":0b011,
    "xori": 0b100,
    "ori":  0b110,
    "andi": 0b111,
    "slli": 0b001,  # shamt (funct7 en imm[11:5]) 0..31
    "srli": 0b101,
    "srai": 0b101,
}

L_FUNCTS = {
    "lb":  0b000,
    "lh":  0b001,
    "lw":  0b010,
    "lbu": 0b100,
    "lhu": 0b101,
}

S_FUNCTS = {
    "sb": 0b000,
    "sh": 0b001,
    "sw": 0b010,
}

B_FUNCTS = {
    "beq": 0b000,
    "bne": 0b001,
    "blt": 0b100,
    "bge": 0b101,
    "bltu":0b110,
    "bgeu":0b111,
}

# -------------------------
# Parsing de líneas
# -------------------------
COMMENT_RE = re.compile(r'(#|//).*')
LABEL_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*):')
WS_COMMA = re.compile(r'\s*,\s*')  # <-- importante para separar operandos por comas

def strip_comment(line: str) -> str:
    m = COMMENT_RE.search(line)
    if m:
        line = line[:m.start()]
    return line.strip()

def split_operands(s: str) -> List[str]:
    if not s: return []
    return WS_COMMA.split(s)

# -------------------------
# Ensamblador de dos pasadas
# -------------------------
class Assembler:
    def __init__(self, lines: List[str]):
        self.lines = lines
        self.symbols: Dict[str, int] = {}
        self.segment = '.text'
        self.addr_text = 0x00000000
        self.addr_data = 0x10000000  # base simbólica para datos (no emite bytes en esta versión)
        self.pc = self.addr_text
        self.insns: List[Tuple[int, str, int]] = []  # (address, line, original_line_no)

    # Pass 1: tabla de símbolos y listado de instrucciones .text
    def pass1(self):
        self.pc = self.addr_text
        self.segment = '.text'
        for idx, raw in enumerate(self.lines, 1):
            line = strip_comment(raw)
            if not line: continue
            # Puede haber múltiples etiquetas seguidas
            while True:
                m = LABEL_RE.match(line)
                if not m: break
                label = m.group(1)
                if label in self.symbols:
                    raise AsmError(f"Línea {idx}: etiqueta duplicada '{label}'")
                self.symbols[label] = self.pc if self.segment == '.text' else self.addr_data
                line = line[m.end():].strip()
                if not line: break
            if not line:
                continue
            # Directivas
            if line.startswith('.'):
                d = line.split()[0]
                if d == '.text':
                    self.segment = '.text'
                elif d == '.data':
                    self.segment = '.data'
                else:
                    # Otras directivas ignoradas en esta versión
                    pass
                continue
            # Instrucciones solo en .text
            if self.segment == '.text':
                self.insns.append((self.pc, line, idx))
                self.pc += 4  # cada instrucción son 4 bytes

    # Pass 2: codificación
    def pass2(self) -> List[int]:
        words: List[int] = []
        i = 0
        while i < len(self.insns):
            addr, line, idx = self.insns[i]
            word = self.encode_line(addr, line, idx, i)
            words.append(word)
            i += 1
        return words

    # Pseudoinstrucciones -> lista de instrucciones base
    def expand_pseudo(self, addr: int, mnem: str, ops: List[str], idx: int) -> List[str]:
        m = mnem.lower()
        if m == 'nop':   return ['addi x0, x0, 0']
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
            # LUI + ADDI con redondeo del bajo
            upper = (imm + (1<<11)) >> 12
            lower = imm - (upper << 12)
            upper &= 0xFFFFF
            return [f"lui {rd}, {upper}", f"addi {rd}, {rd}, {lower}"]
        return [mnem + (' ' + ', '.join(ops) if ops else '')]

    # Codificar una línea (después de expandir pseudo)
    def encode_line(self, addr: int, line: str, idx: int, insn_index: int) -> int:
        parts = line.split(None, 1)
        if not parts:
            raise AsmError(f"Línea {idx}: vacía inesperadamente")
        mnem = parts[0].lower()
        ops = split_operands(parts[1]) if len(parts)>1 else []

        expanded = self.expand_pseudo(addr, mnem, ops, idx)
        if len(expanded) > 1:
            # Insertar el resto de instrucciones justo después y ajustar direcciones/etiquetas
            first = expanded[0]; rest = expanded[1:]
            cur_addr = addr + 4
            insert_at = insn_index + 1
            for text in rest:
                self.insns.insert(insert_at, (cur_addr, text, idx))
                insert_at += 1
                cur_addr += 4
            # Shift de direcciones siguientes
            for j in range(insert_at, len(self.insns)):
                a,l,i2 = self.insns[j]
                self.insns[j] = (a + 4*len(rest), l, i2)
            # Shift de etiquetas posteriores
            for k,v in list(self.symbols.items()):
                if v > addr:
                    self.symbols[k] = v + 4*len(rest)
            # Codificar la primera
            return self.encode_line(addr, first, idx, insn_index)

        # Recalcular mnem/ops ya canónicas
        parts = expanded[0].split(None, 1)
        mnem = parts[0].lower()
        ops = split_operands(parts[1]) if len(parts)>1 else []

        if mnem.startswith('.'):  # no debería llegar aquí
            return 0

        # ---- R-type
        if mnem in R_FUNCTS:
            if len(ops)!=3: raise AsmError(f"Línea {idx}: {mnem} espera 3 operandos (rd, rs1, rs2)")
            rd, rs1, rs2 = parse_reg(ops[0]), parse_reg(ops[1]), parse_reg(ops[2])
            funct7, funct3 = R_FUNCTS[mnem]
            return enc_R(funct7, rs2, rs1, funct3, rd, OP_R)

        # ---- I-type aritmético/lógico (incluye shifts con shamt)
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
                rd, rs1, imm = parse_reg(ops[0]), parse_reg(ops[1]), parse_imm(ops[2])
                if not imm_in_range(imm, 12, signed=True):
                    raise AsmError(f"Línea {idx}: inmediato fuera de rango para {mnem} (12 bits signed)")
                return enc_I(imm, rs1, f3, rd, OP_I)

        # ---- Loads (I-type con offset(rs1))
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

        # ---- Stores (S-type con offset(rs1))
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

        # ---- Branches (B-type con label)
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

        # ---- Jumps
        if mnem == 'jal':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: jal espera 2 operandos (rd, label)")
            rd = parse_reg(ops[0])
            label = ops[1]
            if label not in self.symbols:
                raise AsmError(f"Línea {idx}: etiqueta no definida '{label}'")
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

        # ---- U-type
        if mnem == 'lui' or mnem == 'auipc':
            if len(ops)!=2: raise AsmError(f"Línea {idx}: {mnem} espera 2 operandos (rd, imm20)")
            rd, imm = parse_reg(ops[0]), parse_imm(ops[1])
            if not imm_in_range(imm, 20, signed=False):
                raise AsmError(f"Línea {idx}: inmediato de {mnem} debe caber en 20 bits sin signo")
            opcode = OP_U_LUI if mnem=='lui' else OP_U_AUIPC
            return enc_U(imm<<12, rd, opcode)

        # ---- SYSTEM mínimos
        if mnem == 'ecall':
            return enc_I(0, 0, 0, 0, OP_SYS)
        if mnem == 'ebreak':
            return enc_I(1, 0, 0, 0, OP_SYS)

        raise AsmError(f"Línea {idx}: instrucción inválida o no soportada: '{mnem}'")

# -------------------------
# CLI
# -------------------------
def main(argv=None):
    if argv is None: argv = sys.argv
    if len(argv) != 4:
        print("Uso: python assembler.py program.asm program.hex program.bin")
        return 2
    in_path, hex_path, bin_path = argv[1], argv[2], argv[3]
    try:
        with open(in_path, 'r', encoding='utf-8') as f:
            src = f.read().splitlines()
    except FileNotFoundError:
        print(f"Archivo no encontrado: {in_path}")
        return 1

    asm = Assembler(src)
    try:
        asm.pass1()
        words = asm.pass2()
    except AsmError as e:
        print(f"Error de ensamblado: {e}")
        return 1

    with open(hex_path, 'w', encoding='utf-8') as f:
        for w in words:
            f.write(to_hex8(w) + "\n")
    with open(bin_path, 'w', encoding='utf-8') as f:
        for w in words:
            f.write(to_bin32(w) + "\n")

    print(f"Ensambla OK: {len(words)} instrucciones -> {hex_path}, {bin_path}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
