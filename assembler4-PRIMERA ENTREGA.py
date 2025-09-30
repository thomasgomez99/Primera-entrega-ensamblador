"""
RV32I
- Soporta RV32I: R, I, S, B, U, J
- Pseudoinstrucciones
- Relocaciones por regex: %hi(label), %lo(label)
- Directivas: .text, .data, .word
"""

from __future__ import annotations     # Permite usar anotaciones de tipos futuras/pospuestas. se guarda en cadena 
import sys, re                         # sys: argumentos del programa; re: expresiones regulares.
from typing import List, Tuple, Dict   # Tipos genéricos para anotaciones (listas, tuplas, diccionarios).

# -----------------------------------------------------------------------------
# Errores
# -----------------------------------------------------------------------------
class AsmError(Exception):           # Excepción personalizada para errores del ensamblador
    pass                             # No añade nada extra, solo define la clase, nombre propios

# -----------------------------------------------------------------------------
# Registros
# -----------------------------------------------------------------------------
REGISTROS_ALIAS: Dict[str,int] = {
    "zero":0,                           # x0  : constante cero
    "ra":1,                             # x1  : dirección de retorno
    "sp":2,                             # x2  : puntero de pila
    "gp":3,                             # x3  : puntero global
    "tp":4,                             # x4  : puntero de hilo
    "t0":5, "t1":6, "t2":7,             # x5..x7 : registros temporales (guardados por el llamador)
    "s0":8, "fp":8,                     # x8  : registro guardado / puntero de marco (guardado por
    "s1":9,                             # x9  : registro guardado (guardado por la función llamada)
    "a0":10, "a1":11,                   # x10..x11: argumentos / valores de retorno (guardados por el llamador)
    "a2":12, "a3":13, "a4":14, "a5":15, "a6":16, "a7":17, # x12..x17: argumentos de función (guardados por el llamador)
    "s2":18, "s3":19, "s4":20, "s5":21, "s6":22, "s7":23, "s8":24, "s9":25, "s10":26, "s11":27, # x18..x27: registros guardados (guardados por la función llamada)
    "t3":28, "t4":29, "t5":30, "t6":31  # x28..x31: registros temporales (guardados por el llamador)
}
#Convertir alias/xN a número de registro
def parse_reg(tok: str) -> int:                 #Pasa  de texto a entero, si el texto no es valido, lanza error
    t = tok.strip().lower()                     # Limpia espacios y pasa a minúsculas
    if t.startswith('x') and t[1:].isdigit():   # Caso 1: formato numérico "xN"
        n = int(t[1:])                          # quieres convertir a entero pero ignorando el primer carácter de t
        if 0 <= n <= 31: return n               #   verifica rango y devuelve N
    if t in REGISTROS_ALIAS: return REGISTROS_ALIAS[t]  # Caso 2: alias ("a0","sp","ra",...)
    raise AsmError(f"Registro inválido: {tok}")         # Si no coincide con nada, error

#texto a inmediato
def parse_imm(s: str) -> int:               #convierte un inmediato en texto "10", "-0x1F", "0b1010", "1_000"
    t = s.strip().lower().replace('_', '')  # Limpia espacios, pasa a minúsculas y quita guiones bajos (permite 1_000)
    neg = False
    if t.startswith('-'):                   # Detecta signo negativo
        neg = True; t = t[1:]
    base = 10                               # Base por defecto: decimal
    if t.startswith('0x'):                  # Prefijo hexadecimal
        base = 16; t = t[2:]
    elif t.startswith('0b'):                # Prefijo binario
        base = 2; t = t[2:]
    elif t.startswith('0o'):                # Prefijo octal
        base = 8; t = t[2:]
    try:
        val = int(t, base)                  # Convierte la parte numérica según la base detectada
    except ValueError:
        raise AsmError(f"Inmediato inválido: {s}") # Si no es válido, lanza error del ensamblador
    return -val if neg else val                    # Aplica el signo si hacía falta

def imm_in_range(val: int, bits: int, signed: bool=True) -> bool:  # ¿val cabe en 'bits'? Con signo por defecto. VAL= valor inmediato
    if signed:                                   # Caso: rango con signo (complemento a dos)
        lo = -(1 << (bits-1))                    # Mínimo representable: -2^(bits-1)
        hi = (1 << (bits-1)) - 1                 # Máximo representable:  2^(bits-1) - 1
        return lo <= val <= hi                   # Devuelve True si val ∈ [lo, hi]
    else:                                        # Caso: rango sin signo
        return 0 <= val <= (1<<bits)-1           # True si val ∈ [0, 2^bits - 1]

#Extender signo, tomar un número en n bits y expandirlo a un entero completo conservando su valor con signo.
def sext(val: int, bits: int) -> int:   #0b0111 (7) MSB=0 → positivo  =7
    mask = (1<<bits) - 1          # Crea una máscara con 'bits' unos. Ej: bits=12 -> 0xFFF
    val &= mask                   # Recorta 'val' a esos 'bits' (descarta bits superiores)
    if val & (1<<(bits-1)):       # Si el bit más significativo (bit de signo) está en 1...
        val -= (1<<bits)          # ...resta 2^bits para obtener el valor negativo correcto
    return val                    # Devuelve el entero ya con signo correcto

# Formateador a HEX de 32 bits, completa en ceros
def to_hex8(word: int) -> str:                 #Devuelve 'word' como hexadecimal de 8 dígitos (32 bits), con ceros a la izquierda.
    return format(word & 0xFFFFFFFF, '08x')    # Enmascara a 32 bits y formatea en hex (8 chars, minúscula)

# Formateador a BIN de 32 bits, completa en ceros
def to_bin32(word: int) -> str:               #Devuelve 'word' como binario de 32 bits, con ceros a la izquierda.
    return format(word & 0xFFFFFFFF, '032b')  # Enmascara a 32 bits y formatea en binario (32 caracteres)

# ===== BIN agrupado por formato (R/I/S/B/U/J) ===== Formatos de instrucción. si cumple
def format_bin_grouped(word: int) -> str:  # Muestra la instrucción (32 bits) separada por campos según su formato RV32I
    def bits(val, hi, lo):
        return (val >> lo) & ((1 << (hi - lo + 1)) - 1)  # Extrae el subcampo [hi:lo] de 'val'

    opcode = bits(word, 6, 0)  # Lee el opcode (bits 6..0) para identificar el tipo (R/I/S/B/U/J)

    # R-type: funct7 | rs2 | rs1 | funct3 | rd | opcode
    if opcode == OP_R:
        f7   = bits(word, 31, 25)   # funct7 (7 bits)
        rs2  = bits(word, 24, 20)   # rs2 (5 bits)
        rs1  = bits(word, 19, 15)   # rs1 (5 bits)
        f3   = bits(word, 14, 12)   # funct3 (3 bits)
        rd   = bits(word, 11, 7)    # rd (5 bits)
        return f"{f7:07b} {rs2:05b} {rs1:05b} {f3:03b} {rd:05b} {opcode:07b}"  # Cadena con campos binarios

    # I-type (arit./lógica, loads, jalr, system, fence): imm[11:0] | rs1 | funct3 | rd | opcode
    if opcode in (OP_I, OP_L, OP_I_JALR, OP_SYS, OP_FENCE):
        imm  = bits(word, 31, 20)   # inmediato de 12 bits
        rs1  = bits(word, 19, 15)   # rs1
        f3   = bits(word, 14, 12)   # funct3
        rd   = bits(word, 11, 7)    # rd
        return f"{imm:012b} {rs1:05b} {f3:03b} {rd:05b} {opcode:07b}"

    # S-type (stores): imm[11:5] | rs2 | rs1 | funct3 | imm[4:0] | opcode
    if opcode == OP_S:
        imm11_5 = bits(word, 31, 25)  # parte alta del inmediato
        rs2     = bits(word, 24, 20)  # rs2
        rs1     = bits(word, 19, 15)  # rs1
        f3      = bits(word, 14, 12)  # funct3
        imm4_0  = bits(word, 11, 7)   # parte baja del inmediato
        return f"{imm11_5:07b} {rs2:05b} {rs1:05b} {f3:03b} {imm4_0:05b} {opcode:07b}"

    # B-type (branches): imm[12] | imm[10:5] | rs2 | rs1 | funct3 | imm[4:1] | imm[11] | opcode
    if opcode == OP_B:
        b12   = bits(word, 31, 31)  # bit 12 del inmediato
        b10_5 = bits(word, 30, 25)  # bits 10..5 del inmediato
        rs2   = bits(word, 24, 20)  # rs2
        rs1   = bits(word, 19, 15)  # rs1
        f3    = bits(word, 14, 12)  # funct3
        b4_1  = bits(word, 11, 8)   # bits 4..1 del inmediato
        b11   = bits(word, 7, 7)    # bit 11 del inmediato
        return f"{b12:01b} {b10_5:06b} {rs2:05b} {rs1:05b} {f3:03b} {b4_1:04b} {b11:01b} {opcode:07b}"

    # U-type (lui/auipc): imm[31:12] | rd | opcode
    if opcode in (OP_U_LUI, OP_U_AUIPC):
        imm31_12 = (word >> 12) & 0xFFFFF  # inmediato de 20 bits (alto)
        rd       = bits(word, 11, 7)       # rd
        return f"{imm31_12:020b} {rd:05b} {opcode:07b}"

    # J-type (jal): imm[20] | imm[10:1] | imm[11] | imm[19:12] | rd | opcode
    if opcode == OP_J:
        j20    = bits(word, 31, 31)  # bit 20 del inmediato
        j10_1  = bits(word, 30, 21)  # bits 10..1 del inmediato
        j11    = bits(word, 20, 20)  # bit 11
        j19_12 = bits(word, 19, 12)  # bits 19..12
        rd     = bits(word, 11, 7)   # rd
        return f"{j20:01b} {j10_1:010b} {j11:01b} {j19_12:08b} {rd:05b} {opcode:07b}"

    return to_bin32(word)  # Si no coincide con ningún formato conocido, devuelve los 32 bits seguidos

# -----------------------------------------------------------------------------
# Encoders por formato, enc_R->  codificar una instrucción de tipo R, genera los espacios en el binario
# -----------------------------------------------------------------------------

def enc_R(funct7, rs2, rs1, funct3, rd, opcode):                       # Codifica una instrucción tipo R (funct7|rs2|rs1|funct3|rd|opcode)
    return ((funct7 & 0x7f) << 25) | ((rs2 & 0x1f) << 20) | ((rs1 & 0x1f) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1f) << 7) | (opcode & 0x7f) # funct7 (7b) a bits 31..25, rs2 (5b) a 24..20, rs1 (5b) a 19..15, funct3 (3b) a 14..12, rd (5b) a 11..7, opcode (7b) a 6..0.             
                      
def enc_I(imm, rs1, funct3, rd, opcode):                               # Codifica una instrucción tipo I (imm|rs1|funct3|rd|opcode)
    imm &= 0xFFF                                                       # Limita imm a 12 bits
    return (imm << 20) | ((rs1 & 0x1f) << 15) | ((funct3 & 0x7) << 12) | ((rd & 0x1f) << 7) | ((funct3 & 0x7) << 12) | ((rd & 0x1f) << 7) | (opcode & 0x7f) # imm[11:0] a 31..20, rs1 (5b) a 19..15, # funct3 (3b) a 14..12, rd (5b) a 11..7,

def enc_S(imm, rs2, rs1, funct3, opcode):                              # Codifica una instrucción tipo S (imm[11:5]|rs2|rs1|f3|imm[4:0]|op)
    imm &= 0xFFF                                                       # Limita imm a 12 bits
    imm11_5 = (imm >> 5) & 0x7F                                        # Parte alta del imm (bits 11..5)
    imm4_0  = imm & 0x1F                                               # Parte baja del imm  (bits 4..0)
    return (imm11_5 << 25) | ((rs2 & 0x1f) << 20) | ((rs1 & 0x1f) << 15) | ((funct3 & 0x7) << 12) | (imm4_0 << 7) | (opcode & 0x7f) # imm[11:5] a 31..25, rs2 a 24..20, rs1 a 19..15, funct3 a 14..12, (imm4_0 << 7) | (opcode & 0x7f)                             # imm[4:0] a 11..7, opcode a 6..0.

def enc_B(imm, rs2, rs1, funct3, opcode):                              # Codifica una instrucción tipo B (branch, imm barajado)
    imm = sext(imm, 13)                                                # Asegura imm con signo en 13 bits (offset/2)
    b12   = (imm >> 12) & 1                                            # imm[12]
    b10_5 = (imm >> 5) & 0x3f                                          # imm[10:5]
    b4_1  = (imm >> 1) & 0xf                                           # imm[4:1]
    b11   = (imm >> 11) & 1                                            # imm[11]
    return (b12 << 31) | (b10_5 << 25) | ((rs2 & 0x1f) << 20) | ((rs1 & 0x1f) << 15) | ((funct3 & 0x7) << 12) | (b4_1 << 8) | (b11 << 7) | (opcode & 0x7f)  # imm[12] a bit31, imm[10:5] a 30..25, # rs2 a 24..20, rs1 a 19..15, funct3 a 14..12, imm[4:1] a 11..8, imm[11] a 7, opcode a 6..0.

def enc_U(imm, rd, opcode):                                            # Codifica una instrucción tipo U (lui/auipc)
    return (imm & 0xFFFFF000) | ((rd & 0x1f) << 7) | (opcode & 0x7f)   # imm[31:12] ya alineado (20 bits) + rd a 11..7 opcode a 6..0.

def enc_J(imm, rd, opcode):                                            # Codifica una instrucción tipo J (jal, imm barajado)
    imm = sext(imm, 21)                                                # Asegura imm con signo en 21 bits (offset/2)
    j20    = (imm >> 20) & 1                                           # imm[20]
    j10_1  = (imm >> 1) & 0x3ff                                        # imm[10:1]
    j11    = (imm >> 11) & 1                                           # imm[11]
    j19_12 = (imm >> 12) & 0xff                                        # imm[19:12]
    return (j20 << 31) | (j19_12 << 12) | (j11 << 20) | (j10_1 << 21) | ((rd & 0x1f) << 7) | (opcode & 0x7f) | (opcode & 0x7f)  # Coloca imm barajado en sus posiciones:  j20@31, j19_12@19..12, j11@20, j10_1@30..21; rd@11..7  opcode a 6..0.

# -----------------------------------------------------------------------------
# Tablas ISA — RV32I base -> Cada instrucción en binario termina en su opcode (7 bits)
# -----------------------------------------------------------------------------
OP_R      = 0b0110011   # Opcode instrucciones tipo R (operaciones entre registros: add, sub, and, or, etc.)
OP_I      = 0b0010011   # Opcode instrucciones tipo I aritméticas con inmediatos (addi, xori, andi, etc.)
OP_L      = 0b0000011   # Opcode instrucciones de carga desde memoria (lb, lh, lw, lbu, lhu)
OP_S      = 0b0100011   # Opcode instrucciones de almacenamiento en memoria (sb, sh, sw)
OP_B      = 0b1100011   # Opcode instrucciones de salto condicional (beq, bne, blt, bge, etc.)
OP_J      = 0b1101111   # Opcode salto incondicional largo (jal)
OP_I_JALR = 0b1100111   # Opcode salto indirecto con registro (jalr)
OP_U_LUI  = 0b0110111   # Opcode cargar inmediato en la parte alta de un registro (lui)
OP_U_AUIPC= 0b0010111   # Opcode sumar un inmediato al PC (auipc)
OP_SYS    = 0b1110011   # Opcode instrucciones del sistema (ecall, ebreak, csr*)
OP_FENCE  = 0b0001111   # Opcode instrucciones de sincronización de memoria (fence, fence.i)

R_FUNCTS = {                             # Mapa de mnemónicos R-type -> (funct7, funct3)
    "add":  (0b0000000, 0b000),          # add  : suma (funct7=0000000, funct3=000)
    "sub":  (0b0100000, 0b000),          # sub  : resta (se diferencia de add por funct7=0100000)
    "sll":  (0b0000000, 0b001),          # sll  : shift lógico izq.      (funct3=001)
    "slt":  (0b0000000, 0b010),          # slt  : set if less-than (signed)
    "sltu": (0b0000000, 0b011),          # sltu : set if less-than (unsigned)
    "xor":  (0b0000000, 0b100),          # xor  : XOR bit a bit
    "srl":  (0b0000000, 0b101),          # srl  : shift lógico der.      (funct7=0000000)
    "sra":  (0b0100000, 0b101),          # sra  : shift aritmético der.  (funct7=0100000)
    "or":   (0b0000000, 0b110),          # or   : OR bit a bit
    "and":  (0b0000000, 0b111),          # and  : AND bit a bit
}

I_FUNCTS = {                                # Mapa de mnemónicos I-type (ALU con inmediato) -> funct3
    "addi": 0b000, "slti": 0b010, "sltiu":0b011,  # suma, set-less-than (con/sin signo)
    "xori": 0b100, "ori":  0b110, "andi": 0b111,  # XOR, OR, AND con inmediato
    "slli": 0b001, "srli": 0b101, "srai": 0b101,  # shifts inmediatos: izq, der lógico y der aritm.
}                                                # (Nota: srli y srai comparten funct3=101; se distinguen por los bits altos del imm/funct7)

L_FUNCTS = {                                # Mapa de LOADs -> funct3
    "lb":  0b000, "lh":  0b001, "lw":  0b010,     # carga con signo: byte, half, word
    "lbu": 0b100, "lhu": 0b101,                   # carga sin signo: byte, half
}

S_FUNCTS = { 
    "sb": 0b000,   # Store Byte (guardar un byte en memoria)
    "sh": 0b001,   # Store Halfword (guardar 16 bits = 2 bytes)
    "sw": 0b010    # Store Word (guardar 32 bits = 4 bytes)
}  # Diccionario que asocia instrucciones de almacenamiento (tipo S) con su campo funct3

B_FUNCTS = {                                # BRANCHes -> funct3
    "beq":0b000, "bne":0b001, "blt":0b100,       # igual, distinto, menor (signed)
    "bge":0b101, "bltu":0b110, "bgeu":0b111,     # mayor/igual (signed), menor/ mayor/igual (unsigned)
}

# -----------------------------------------------------------------------------
# Parsing de líneas (expresiones regulares)
# -----------------------------------------------------------------------------
#borrar/ignorar la parte comentada de una línea.
COMMENT_RE = re.compile(r'(#|//).*')                 # Expresión regular para comentarios: empieza con '#' o '//' y sigue hasta fin de línea
#construir la tabla de símbolos en el primer pase.
LABEL_RE   = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*):')# Expresión regular para etiquetas al inicio: nombre válido seguido de ':' (captura el nombre)
#una coma con espacios opcionales alrededor.
WS_COMMA   = re.compile(r'\s*,\s*')                   # Expresión regular para coma con espacios opcionales alrededor (útil para separar operandos)


# Relocalizaciones estilo RISC-V: patrones para reconocer %hi(label) y %lo(label)
RE_HI = re.compile(r'^\s*%hi\(([A-Za-z_][A-Za-z0-9_]*)\)\s*$')  # Coincide con toda la línea: opc. espacios, %hi(nombre), opc. espacios. Captura 'nombre'
RE_LO = re.compile(r'^\s*%lo\(([A-Za-z_][A-Za-z0-9_]*)\)\s*$')  # Igual que arriba pero para %lo(nombre); útil para LUI/ADDI con direcciones de etiquetas


def eval_reloc(expr: str, symbols: Dict[str,int]) -> int:   # Define la función que recibe una expresión y la tabla de símbolos
    """
    Evalúa %hi(label) y %lo(label) usando la tabla de símbolos.
    - %hi(label) = (addr(label) + 0x800) >> 12
    - %lo(label) = addr(label) & 0xFFF   (el signo lo maneja la I-type)
    Lanza ValueError si 'expr' no es una reloc %hi/%lo.
    """
    m = RE_HI.match(expr)                                  # Intenta reconocer si la expresión es %hi(...)
    if m:                                                  # Si hubo coincidencia con el patrón %hi
        lab = m.group(1)                                   # Extrae el nombre de la etiqueta dentro de %hi(...)
        if lab not in symbols:                             # Verifica que la etiqueta exista en la tabla de símbolos
            raise AsmError(f"Etiqueta no definida en %hi(): '{lab}'")  # Si no existe, lanza error de ensamblador
        addr = symbols[lab]                                # Obtiene la dirección asociada a la etiqueta
        return (addr + 0x800) >> 12                        # Devuelve los 20 bits altos con redondeo (se desplaza 12)

    m = RE_LO.match(expr)                                  # Intenta reconocer si la expresión es %lo(...)
    if m:                                                  # Si hubo coincidencia con el patrón %lo
        lab = m.group(1)                                   # Extrae el nombre de la etiqueta dentro de %lo(...)
        if lab not in symbols:                             # Verifica que la etiqueta exista en la tabla de símbolos
            raise AsmError(f"Etiqueta no definida en %lo(): '{lab}'")  # Si no existe, lanza error de ensamblador
        addr = symbols[lab]                                # Obtiene la dirección asociada a la etiqueta
        return addr & 0xFFF                                # Devuelve los 12 bits bajos (parte baja de la dirección)

    raise ValueError("no relocation")                      # Si no es ni %hi ni %lo, lanza error genérico

# elimina el comentario de una línea y recorta espacios en los extremos.
def strip_comment(line: str) -> str:                 # Define la función que recibe una línea de texto
    m = COMMENT_RE.search(line)                      # Busca un comentario usando la regex COMMENT_RE
    if m: line = line[:m.start()]                    # Si encontró, corta la línea antes del inicio del comentario
    return line.strip()                              # Devuelve la línea sin comentario y sin espacios en los extremos

# separa los operandos de una instrucción por comas o espacios.
def split_operands(s: str) -> List[str]:             # Define la función que recibe la cadena de operandos
    return [] if not s else WS_COMMA.split(s)        # Si s está vacío, devuelve []; si no, divide por regex WS_COMMA

# -----------------------------------------------------------------------------
# Ensamblador de 2 pasadas
# -----------------------------------------------------------------------------
class Assembler:                                      # Define una clase (tipo de objeto) llamada Assembler
    def __init__(self, lines: List[str]):            # Constructor: se llama al crear Assembler(...); recibe líneas de código fuente
        self.lines   = lines                          # Guarda las líneas de entrada (el código ensamblador)
        self.symbols: Dict[str,int] = {}              # Tabla de símbolos: etiqueta -> dirección (vacía al inicio)
        self.segment = '.text'                        # Segmento actual por defecto (código)
        self.addr_text = 0x00000000                   # Dirección base para el segmento .text
        self.addr_data = 0x10000000  # base para .data  # Dirección base para el segmento .data
        self.pc = self.addr_text                      # Contador de programa (PC) comienza en la base de .text
        self.insns: List[Tuple[int,str,int]] = []     # Lista de instrucciones parseadas: (addr, texto, nº de línea)
        self.data_words: List[Tuple[int,int]] = []    # Palabras de datos: (addr, valor) para el segmento .data

    def pass1(self):                                  # Define la primera pasada del ensamblador (construye símbolos/índices)
        self.pc = self.addr_text                      # Inicializa el contador de programa (PC) a la base del segmento .text
        self.segment = '.text'                        # Fija el segmento activo inicial a .text (código)
        for idx, raw in enumerate(self.lines, 1):     # Recorre cada línea del fuente con número de línea desde 1
            line = strip_comment(raw)                 # Elimina comentarios y espacios sobrantes de la línea
            if not line: continue                     # Si la línea queda vacía tras limpiar, salta a la siguiente

            # Etiquetas (pueden venir varias seguidas)
            while True:                                           # Bucle por si hay varias etiquetas consecutivas en la misma línea
                m = LABEL_RE.match(line)                          # Intenta casar una etiqueta al inicio de la línea con la regex LABEL_RE
                if not m: break                                   # Si no hay coincidencia, sale del bucle de etiquetas
                label = m.group(1)                                # Extrae el nombre de la etiqueta capturada por la regex
                if label in self.symbols:                         # Verifica si la etiqueta ya fue definida antes
                    raise AsmError(f"Línea {idx}: etiqueta duplicada '{label}'")  # Lanza error por etiqueta duplicada
                # Dirección según segmento activo
                if self.segment == '.text':                       # Si el segmento actual es .text (código)
                    self.symbols[label] = self.pc                 # Asigna a la etiqueta la dirección actual del PC
                else:                                             # En otro caso (segmento .data)
                    self.symbols[label] = self.addr_data          # Asigna a la etiqueta la dirección actual de datos
                line = line[m.end():].strip()                     # Corta la etiqueta consumida del inicio y recorta espacios
                if not line: break                                # Si ya no queda nada en la línea, termina el bucle de etiquetas
            if not line: continue                                 # Si la línea quedó vacía tras procesar etiquetas, pasa a la siguiente

            # Directivas
            if line.startswith('.'):                                  # Si la línea comienza con '.', se trata de una directiva
                parts = line.split()                                  # Separa la directiva del resto de argumentos por espacios
                d = parts[0].lower()                                  # Obtiene el nombre de la directiva en minúsculas (p. ej., '.text')
                if d == '.text':                                      # Si es la directiva .text...
                    self.segment = '.text'                            # ...cambia el segmento activo al de código (.text)
                elif d == '.data':                                    # Si es la directiva .data...
                    self.segment = '.data'                            # ...cambia el segmento activo al de datos (.data)
                elif d == '.word' and self.segment == '.data':        # Si es .word y estamos en el segmento de datos...
                    if len(parts) < 2:                                # Debe haber al menos un operando tras .word
                        raise AsmError(f"Línea {idx}: .word requiere al menos un operando")  # Error si faltan operandos
                    # Aceptamos lista: .word 1, 2, 0x33
                    payload = ' '.join(parts[1:])                     # Junta los argumentos restantes como una cadena
                    for tok in WS_COMMA.split(payload):               # Divide por comas/espacios usando la regex WS_COMMA
                        val = parse_imm(tok)                          # Convierte cada token a entero (decimal/hex, etc.)
                        self.data_words.append((self.addr_data, val)) # Registra (dirección de datos actual, valor) en la lista de datos
                        self.addr_data += 4                           # Avanza el puntero de datos 4 bytes (tamaño de una palabra)
                # (Añade aquí .byte/.half si lo necesitas)            # Punto de extensión para otras directivas de datos
                continue                                              # Pasa a la siguiente línea del fuente (ya manejamos esta)

            # Instrucciones (solo en .text)
            if self.segment == '.text':                               # Solo registramos instrucciones cuando el segmento activo es .text
                self.insns.append((self.pc, line, idx))               # Guarda (dirección actual, texto de instrucción, número de línea)
                self.pc += 4                                          # Avanza el PC asumiendo instrucciones de 4 bytes de longitud

    def pass2(self) -> List[int]:                        # Segunda pasada: codifica cada instrucción y devuelve la lista de palabras
        words: List[int] = []                            # Acumulará las palabras máquina (enteros de 32 bits)
        i = 0                                            # Índice para recorrer la lista de instrucciones self.insns
        while i < len(self.insns):                       # Itera hasta procesar todas las instrucciones recogidas en pass1
            addr, line, idx = self.insns[i]              # Desempaqueta: dirección, texto de la instrucción y nº de línea original
            w = self.encode_line(addr, line, idx, i)     # Codifica la instrucción en una palabra máquina llamando al encoder
            words.append(w)                              # Añade la palabra resultante a la salida
            i += 1                                       # Avanza al siguiente elemento
        return words                                     # Devuelve la lista completa de palabras codificadas

    # ---------------- Pseudoinstrucciones ----------------

    def expand_pseudo(self, addr: int, mnem: str, ops: List[str], idx: int) -> List[str]:  # Expande seudoinstrucciones a instrucciones reales (RISC-V)
        m = mnem.lower()                                                                 # Normaliza el mnemónico a minúsculas
        if m == 'nop':  return ['addi x0, x0, 0']                                        # nop ≡ addi x0, x0, 0

        if m == 'mv':                                                                    # mv rd, rs ≡ addi rd, rs, 0
            if len(ops)!=2: raise AsmError(f"Línea {idx}: mv espera 2 operandos")        # Valida cantidad de operandos
            return [f"addi {ops[0]}, {ops[1]}, 0"]                                       # Emite instrucción equivalente

        if m == 'not':                                                                   # not rd, rs ≡ xori rd, rs, -1
            if len(ops)!=2: raise AsmError(f"Línea {idx}: not espera 2 operandos")       # Valida cantidad de operandos
            return [f"xori {ops[0]}, {ops[1]}, -1"]                                      # Emite instrucción equivalente

        if m == 'neg':                                                                   # neg rd, rs ≡ sub rd, x0, rs
            if len(ops)!=2: raise AsmError(f"Línea {idx}: neg espera 2 operandos")       # Valida cantidad de operandos
            return [f"sub {ops[0]}, x0, {ops[1]}"]                                       # Emite instrucción equivalente

        if m == 'seqz':                                                                  # seqz rd, rs ≡ sltiu rd, rs, 1  (rd=1 si rs==0)
            if len(ops)!=2: raise AsmError(f"Línea {idx}: seqz espera 2 operandos")      # Valida cantidad de operandos
            return [f"sltiu {ops[0]}, {ops[1]}, 1"]                                      # Emite instrucción equivalente

        if m == 'snez':                                                                  # snez rd, rs ≡ sltu rd, x0, rs  (rd=1 si rs!=0)
            if len(ops)!=2: raise AsmError(f"Línea {idx}: snez espera 2 operandos")      # Valida cantidad de operandos
            return [f"sltu {ops[0]}, x0, {ops[1]}"]                                      # Emite instrucción equivalente

        if m == 'sltz':                                                                  # sltz rd, rs ≡ slt rd, rs, x0   (rd=1 si rs<0)
            if len(ops)!=2: raise AsmError(f"Línea {idx}: sltz espera 2 operandos")      # Valida cantidad de operandos
            return [f"slt {ops[0]}, {ops[1]}, x0"]                                       # Emite instrucción equivalente

        if m == 'sgtz':                                                                  # sgtz rd, rs ≡ slt rd, x0, rs   (rd=1 si rs>0)
            if len(ops)!=2: raise AsmError(f"Línea {idx}: sgtz espera 2 operandos")      # Valida cantidad de operandos
            return [f"slt {ops[0]}, x0, {ops[1]}"]                                       # Emite instrucción equivalente

        if m == 'j':                                                                     # j label ≡ jal x0, label  (salto sin enlazar)
            if len(ops)!=1: raise AsmError(f"Línea {idx}: j espera etiqueta")            # Debe recibir 1 operando (etiqueta)
            return [f"jal x0, {ops[0]}"]                                                 # Emite instrucción equivalente

        if m == 'jr':                                                                    # jr rs ≡ jalr x0, rs, 0   (salto indirecto)
            if len(ops)!=1: raise AsmError(f"Línea {idx}: jr espera registro")           # Debe recibir 1 operando (registro)
            return [f"jalr x0, {ops[0]}, 0"]                                             # Emite instrucción equivalente

        if m == 'ret':                                                                   # ret ≡ jalr x0, ra, 0     (retorno)
            if len(ops)!=0: raise AsmError(f"Línea {idx}: ret no lleva operandos")       # No debe llevar operandos
            return ["jalr x0, ra, 0"]                                                    # Emite instrucción equivalente

        if m in ('beqz','bnez'):                                                         # beqz/bnez rs, label → beq/bne rs, x0, label
            if len(ops)!=2: raise AsmError(f"Línea {idx}: {m} espera 2 operandos (rs, label)")  # Valida operandos
            cond = 'beq' if m=='beqz' else 'bne'                                         # Selecciona condición (igual o distinto a cero)
            return [f"{cond} {ops[0]}, x0, {ops[1]}"]                                    # Emite instrucción equivalente

        if m == 'li':                                                                    # li rd, imm → addi (si cabe) o lui+addi (si no)
            if len(ops)!=2: raise AsmError(f"Línea {idx}: li espera 2 operandos")        # Valida cantidad de operandos
            rd = ops[0]; imm = parse_imm(ops[1])                                         # rd destino y convierte inmediato textual a entero
            if imm_in_range(imm, 12, signed=True):                                       # ¿Inmediato cabe en 12 bits con signo?
                return [f"addi {rd}, x0, {imm}"]                                         # Sí: usar addi con x0
            upper = (imm + (1<<11)) >> 12                                                # No: calcula parte alta (con redondeo de bit 11)
            lower = imm - (upper << 12)                                                  # Parte baja = imm - (upper << 12)
            upper &= 0xFFFFF                                                             # Asegura upper en 20 bits
            return [f"lui {rd}, {upper}", f"addi {rd}, {rd}, {lower}"]                   # Emite LUI seguido de ADDI

        return [mnem + (' ' + ', '.join(ops) if ops else '')]                             # Si no es pseudo, devuelve la instrucción original


    # ---------------- Codificación ----------------

    def encode_line(self, addr: int, line: str, idx: int, insn_index: int) -> int:  # Codifica una línea a palabra máquina (32 bits)
        parts = line.split(None, 1)                                             # Separa mnemónico y resto (operandos) por primer espacio
        if not parts:                                                           # Si no hay nada tras limpiar...
            raise AsmError(f"Línea {idx}: vacía inesperadamente")               # ...lanza error de línea vacía
        mnem = parts[0].lower()                                                 # Mnemónico en minúsculas
        ops  = split_operands(parts[1]) if len(parts)>1 else []                 # Lista de operandos (o vacía si no hay)

        # Expansión de pseudo
        expanded = self.expand_pseudo(addr, mnem, ops, idx)                     # Expande seudoinstrucciones a reales (puede devolver varias)
        if len(expanded) > 1:                                                   # Si la expansión produjo más de una instrucción...
            first, rest = expanded[0], expanded[1:]                              # Separa la primera de las restantes
            cur_addr = addr + 4                                                 # Dirección de la siguiente instrucción
            insert_at = insn_index + 1                                          # Posición para insertar instrucciones extra en self.insns
            for text in rest:                                                   # Inserta las instrucciones restantes con sus direcciones
                self.insns.insert(insert_at, (cur_addr, text, idx))             # Inserta (addr, texto, nº de línea)
                insert_at += 1; cur_addr += 4                                   # Avanza posición de inserción y dirección
            # Reajusta direcciones posteriores
            for j in range(insert_at, len(self.insns)):                         # Recorre las instrucciones siguientes
                a,l,i2 = self.insns[j]                                          # Desempaqueta la tupla
                self.insns[j] = (a + 4*len(rest), l, i2)                        # Desplaza sus direcciones por el tamaño insertado
            for k,v in list(self.symbols.items()):                              # Recorre símbolos (etiquetas)
                if v > addr:                                                    # Si apuntan a direcciones después de la actual...
                    self.symbols[k] = v + 4*len(rest)                           # ...ajusta la dirección del símbolo
            return self.encode_line(addr, first, idx, insn_index)               # Re-codifica la primera (ya sin pseudo), recursivamente

        parts = expanded[0].split(None, 1)                                      # Toma la instrucción expandida (única) y separa mnem/ops
        mnem = parts[0].lower()                                                 # Mnemónico normalizado
        ops  = split_operands(parts[1]) if len(parts)>1 else []                 # Operandos de la instrucción expandida

        if mnem.startswith('.'):                                                # Si comienza con '.', es directiva (no se codifica)
            return 0                                                            # Devuelve 0 como relleno/no-op para directivas

        # ----- R-type -----
        if mnem in R_FUNCTS:                                                    # Si es instrucción tipo R (add, sub, and, or, etc.)
            if len(ops)!=3: raise AsmError(f"Línea {idx}: {mnem} espera 3 operandos (rd, rs1, rs2)")  # Valida operandos
            rd, rs1, rs2 = parse_reg(ops[0]), parse_reg(ops[1]), parse_reg(ops[2])  # Convierte registros
            funct7, funct3 = R_FUNCTS[mnem]                                     # Busca funct7/funct3 del mnemónico
            return enc_R(funct7, rs2, rs1, funct3, rd, OP_R)                    # Codifica formato R

        # ----- I-type aritm/log (incluye shifts) -----
        if mnem in I_FUNCTS:                                                    # Instrucción tipo I aritmética/lógica
            f3 = I_FUNCTS[mnem]                                                 # funct3 correspondiente
            if mnem in ('slli','srli','srai'):                                  # Caso especial: shifts con shamt codificado en imm
                if len(ops)!=3: raise AsmError(f"Línea {idx}: {mnem} espera 3 operandos (rd, rs1, shamt)")  # Valida
                rd, rs1, sh = parse_reg(ops[0]), parse_reg(ops[1]), parse_imm(ops[2])  # Lee rd, rs1 y el desplazamiento
                if not imm_in_range(sh, 5, signed=False):                       # Verifica que shamt esté en 0..31 (5 bits)
                    raise AsmError(f"Línea {idx}: desplazamiento fuera de rango (0..31)")  # Error si no cabe
                funct7 = 0b0100000 if mnem=='srai' else 0b0000000               # srai usa funct7=0x20; slli/srli usan 0x00
                imm = (funct7<<5) | (sh & 0x1f)                                 # Construye imm con funct7:shamt
                return enc_I(imm, rs1, f3, rd, OP_I)                            # Codifica formato I
            else:                                                               # Resto de tipo I (addi, xori, ori, etc.)
                if len(ops)!=3: raise AsmError(f"Línea {idx}: {mnem} espera 3 operandos (rd, rs1, imm)")  # Valida
                rd, rs1 = parse_reg(ops[0]), parse_reg(ops[1])                  # Convierte registros
                # Reloc %lo(label) o inmediato numérico
                try:
                    imm = eval_reloc(ops[2], self.symbols)                      # Intenta evaluar %lo(label)
                except ValueError:
                    imm = parse_imm(ops[2])                                     # Si no, parsea inmediato numérico
                if not imm_in_range(imm, 12, signed=True):                      # Verifica rango de 12 bits con signo
                    raise AsmError(f"Línea {idx}: inmediato fuera de rango para {mnem} (12 bits signed)")  # Error
                return enc_I(imm, rs1, f3, rd, OP_I)                            # Codifica formato I

        # ----- Loads (I-type) -----
        if mnem in L_FUNCTS:                                                    # Cargas (lb, lh, lw, lbu, lhu, etc.)
            if len(ops)!=2: raise AsmError(f"Línea {idx}: {mnem} espera 2 operandos (rd, offset(rs1))")  # Valida
            rd = parse_reg(ops[0])                                              # Registro destino
            m = re.match(r'^(-?0x[0-9a-fA-F]+|-?\d+)\(([^)]+)\)$', ops[1].replace(' ',''))  # Matchea offset(rs1)
            if not m: raise AsmError(f"Línea {idx}: formato de dirección inválido, use offset(rs1)")     # Error si no coincide
            imm = parse_imm(m.group(1))                                         # Convierte offset
            rs1 = parse_reg(m.group(2))                                         # Convierte rs1
            if not imm_in_range(imm, 12, signed=True):                          # Rango de 12 bits con signo
                raise AsmError(f"Línea {idx}: inmediato fuera de rango (12 bits)")  # Error si no cabe
            return enc_I(imm, rs1, L_FUNCTS[mnem], rd, OP_L)                    # Codifica load (I-type con opcode de load)

        # ----- Stores (S-type) -----
        if mnem in S_FUNCTS:                                                    # Almacenamientos (sb, sh, sw)
            if len(ops)!=2: raise AsmError(f"Línea {idx}: {mnem} espera 2 operandos (rs2, offset(rs1))")  # Valida
            rs2 = parse_reg(ops[0])                                             # Registro fuente a guardar
            m = re.match(r'^(-?0x[0-9a-fA-F]+|-?\d+)\(([^)]+)\)$', ops[1].replace(' ',''))  # Matchea offset(rs1)
            if not m: raise AsmError(f"Línea {idx}: formato de dirección inválido, use offset(rs1)")     # Error si no coincide
            imm = parse_imm(m.group(1))                                         # Convierte offset
            rs1 = parse_reg(m.group(2))                                         # Convierte rs1
            if not imm_in_range(imm, 12, signed=True):                          # Rango de 12 bits con signo
                raise AsmError(f"Línea {idx}: inmediato fuera de rango (12 bits)")  # Error si no cabe
            return enc_S(imm, rs2, rs1, S_FUNCTS[mnem], OP_S)                   # Codifica store (S-type)

        # ----- Branches (B-type) -----
        if mnem in B_FUNCTS:                                                    # Saltos condicionales (beq, bne, blt, bge, etc.)
            if len(ops)!=3: raise AsmError(f"Línea {idx}: {mnem} espera 3 operandos (rs1, rs2, label)")  # Valida
            rs1, rs2 = parse_reg(ops[0]), parse_reg(ops[1])                     # Convierte registros
            label = ops[2]                                                      # Etiqueta destino
            if label not in self.symbols:                                       # Verifica que la etiqueta exista
                raise AsmError(f"Línea {idx}: etiqueta no definida '{label}'")  # Error si no existe
            target = self.symbols[label]                                        # Dirección de la etiqueta
            offset = target - addr                                              # Desplazamiento relativo a la instrucción
            if not imm_in_range(offset, 13, signed=True):                       # Rango de 13 bits con signo (B-type)
                raise AsmError(f"Línea {idx}: desplazamiento de branch fuera de rango")  # Error si no cabe
            return enc_B(offset, rs2, rs1, B_FUNCTS[mnem], OP_B)                # Codifica branch (B-type)

        # ----- Jumps (J-type y JALR) -----
        if mnem == 'jal':                                                       # Salto absoluto con link
            if len(ops)!=2: raise AsmError(f"Línea {idx}: jal espera 2 operandos (rd, label)")  # Valida
            rd = parse_reg(ops[0]); label = ops[1]                              # Convierte rd y toma etiqueta
            if label not in self.symbols: raise AsmError(f"Línea {idx}: etiqueta no definida '{label}'")  # Verifica etiqueta
            target = self.symbols[label]                                        # Dirección de la etiqueta
            offset = target - addr                                              # Desplazamiento relativo
            if not imm_in_range(offset, 21, signed=True):                       # Rango de 21 bits con signo (J-type)
                raise AsmError(f"Línea {idx}: desplazamiento de JAL fuera de rango")  # Error si no cabe
            return enc_J(offset, rd, OP_J)                                      # Codifica JAL (J-type)

        if mnem == 'jalr':                                                      # Salto indirecto con link
            if len(ops)!=3: raise AsmError(f"Línea {idx}: jalr espera 3 operandos (rd, rs1, imm)")  # Valida
            rd, rs1, imm = parse_reg(ops[0]), parse_reg(ops[1]), parse_imm(ops[2])  # Convierte rd, rs1 y el inmediato
            if not imm_in_range(imm, 12, signed=True):                          # Rango 12 bits con signo
                raise AsmError(f"Línea {idx}: inmediato fuera de rango (12 bits)")  # Error si no cabe
            return enc_I(imm, rs1, 0b000, rd, OP_I_JALR)                        # Codifica JALR (I-type con funct3=000)

        # ----- U-type (LUI/AUIPC) -----
        if mnem == 'lui' or mnem == 'auipc':                                    # Instrucciones de 20 bits superiores
            if len(ops)!=2: raise AsmError(f"Línea {idx}: {mnem} espera 2 operandos (rd, imm20)")  # Valida
            rd = parse_reg(ops[0])                                              # Convierte rd
            # Reloc %hi(label) o inmediato numérico
            try:
                imm20 = eval_reloc(ops[1], self.symbols)                        # Intenta evaluar %hi(label)
            except ValueError:
                imm20 = parse_imm(ops[1])                                       # Si no, parsea inmediato numérico
            if not imm_in_range(imm20, 20, signed=False):                       # Debe caber en 20 bits sin signo
                raise AsmError(f"Línea {idx}: inmediato de {mnem} debe caber en 20 bits sin signo")  # Error
            opcode = OP_U_LUI if mnem=='lui' else OP_U_AUIPC                    # Selecciona opcode según mnemónico
            return enc_U(imm20<<12, rd, opcode)                                 # Codifica U-type (imm alineado a 12 bits)

        # ----- SYSTEM mínimos -----
        if mnem == 'ecall':  return enc_I(0, 0, 0, 0, OP_SYS)                   # ecall: codifica instrucción de sistema
        if mnem == 'ebreak': return enc_I(1, 0, 0, 0, OP_SYS)                   # ebreak: codifica instrucción de breakpoint

        raise AsmError(f"Línea {idx}: instrucción inválida o no soportada: '{mnem}'")  # Si nada aplica, error de instrucción


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv=None):                                                                  # Punto de entrada principal del programa
    if argv is None: argv = sys.argv                                                  # Si no se pasa argv, usa los argumentos de la línea de comandos
    if len(argv) != 4:                                                                # Se esperan exactamente 3 argumentos: in, out_hex, out_bin
        print("Uso: python assembler2.02.py program.asm program.hex program.bin")     # Mensaje de uso si falta/sobra algo
        return 2                                                                      # Código de salida 2: uso incorrecto
    in_path, hex_path, bin_path = argv[1], argv[2], argv[3]                           # Desempaqueta rutas: entrada .asm, salida .hex, salida .bin

    try:
        with open(in_path, 'r', encoding='utf-8') as f:                               # Abre el archivo de entrada en modo lectura UTF-8
            lines = f.read().splitlines()                                             # Lee todo y separa por líneas (sin saltos de línea)
    except FileNotFoundError:                                                          # Si el archivo no existe...
        print(f"Archivo no encontrado: {in_path}")                                    # Informa ruta no encontrada
        return 1                                                                      # Código de salida 1: error

    asm = Assembler(lines)                                                            # Crea una instancia del ensamblador con las líneas leídas
    try:
        asm.pass1()                                                                   # Primera pasada: construye símbolos, separa .text/.data
        words_text = asm.pass2()        # instrucciones .text                         # Segunda pasada: codifica instrucciones a palabras
    except AsmError as e:                                                             # Captura errores de ensamblado (propios del parser/encoder)
        print(f"Error de ensamblado: {e}")                                            # Muestra el mensaje de error al usuario
        return 1                                                                      # Código de salida 1: error

    # Escribir salidas separadas
    with open(hex_path, 'w', encoding='utf-8') as fhex, open(bin_path, 'w', encoding='utf-8') as fbin:  # Abre salidas .hex y .bin
        # --- .text ---
        fhex.write("# .text\n")                                                       # Encabezado de sección .text en el .hex (comentario)
        fbin.write("# .text\n")                                                       # Encabezado de sección .text en el .bin (comentario)
        for w in words_text:                                                          # Recorre cada palabra de código máquina (.text)
            fhex.write(to_hex8(w) + '\n')                                             # Escribe la palabra en formato hexadecimal de 8 dígitos
            fbin.write(format_bin_grouped(w) + '\n')                                  # Escribe la palabra en binario agrupado (legible)

        # --- .data ---
        fhex.write("# .data\n")                                                       # Encabezado de sección .data en el .hex
        fbin.write("# .data\n")                                                       # Encabezado de sección .data en el .bin
        for addr, val in asm.data_words:                                              # Recorre las palabras de datos generadas en pass1
            # Los datos se escriben tal cual (word de 32 bits)
            fhex.write(to_hex8(val) + '\n')                                           # Escribe el dato en hexadecimal de 8 dígitos
            fbin.write(format_bin_grouped(val) + '\n')                                # Escribe el dato en binario agrupado

    print(f"Ensambla OK:")                                                            # Mensaje final de éxito
    print(f"  .text: {len(words_text)} palabras")                                     # Reporta cuántas palabras se generaron en .text
    print(f"  .data: {len(asm.data_words)} palabras")                                 # Reporta cuántas palabras hay en .data
    print(f"Salidas -> {hex_path}, {bin_path}")                                       # Muestra rutas de archivos de salida
    return 0                                                                          # Código de salida 0: ejecución exitosa

if __name__ == '__main__':                                                            # Si el módulo se ejecuta directamente (no importado)...
    sys.exit(main())                                                                  # Llama a main() y sale con su código de retorno

