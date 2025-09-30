
# ============================================================
# TEST SUITE: Restricciones y Errores del Ensamblador RV32I
# Uso: descomenta UN solo bloque de prueba a la vez y ensambla:
#   python assembler.py restricciones_suite.asm out.hex out.bin
# ============================================================

.text
start:
    # ---- BASE VÁLIDA (compila) ----
    addi  a0, x0, 10        # I ok
    addi  a1, a0, -5        # I ok (signed)
    add   a2, a0, a1        # R ok
    slli  a3, a2, 2         # I shift ok (0..31)
    lui   t0, 0x12345       # U ok (20 bits)
    auipc t1, 0x1           # U ok
    sw    a2, 0(sp)         # S ok
    beq   a2, a2, done      # B ok
    addi  a4, x0, 999       # no se ejecuta si el branch salta
done:
    ret                     # pseudo → jalr x0, ra, 0


# ============================================================
# RESTRICCIÓN 1: Error de SINTAXIS
# - Quita el '#' de UNA línea para ver el error correspondiente
# ============================================================
# .textx                       # ❌ directiva mal escrita
# addi a0 x0, 1                # ❌ falta coma entre operandos
# lw a0, 0x10 x1               # ❌ falta el paréntesis en offset(rs1)
# beq a0, a1 done              # ❌ falta coma antes de label


# ============================================================
# RESTRICCIÓN 2: Instrucción INVÁLIDA (mnemonic desconocido)
# ============================================================
# addx  a0, a1, a2             # ❌ 'addx' no existe en RV32I
# mov   a0, a1                  # ❌ usa 'mv' (pseudo), no 'mov'


# ============================================================
# RESTRICCIÓN 3: Operandos INCORRECTOS (número/tipo)
# ============================================================
# addi a0, a1, a2               # ❌ addi espera inmediato, no registro
# add  a0, a1                   # ❌ faltan operandos (R-type requiere 3)
# slli a0, a1, 40               # ❌ shamt fuera de 0..31 (ver Res. 5 para rango)
# lw   a0, a1                   # ❌ loads deben ser rd, offset(rs1)


# ============================================================
# RESTRICCIÓN 4: Etiqueta NO DEFINIDA
# ============================================================
# beq a0, a1, etiqueta_que_no_existe   # ❌ label indefinida
# jal ra, mas_alla                      # ❌ otra referencia indefinida


# ============================================================
# RESTRICCIÓN 5: Inmediato FUERA DE RANGO
# (lo que excede los bits permitidos por formato)
# ============================================================
# addi a0, a1, 4096             # ❌ I aritmético: ±2047 máx (12 bits signed)
# addi a0, a1, -4097            # ❌ fuera de rango negative
# slli a0, a1, 32               # ❌ shamt en RV32: 0..31
# lw   a0, 3000(a1)             # ❌ offset de 12 bits signed (−2048..2047)
# sw   a2, -3000(a1)            # ❌ offset fuera de rango
# lui  a0, 0x200000             # ❌ U-type: inmediato debe caber en 20 bits (0..0xFFFFF)


# ============================================================
# RESTRICCIÓN 6: Formato de LOAD/STORE inválido
# ============================================================
# lw a0, (a1)                   # ❌ falta offset numérico: usa 0(a1)
# lw a0, 0x10                   # ❌ falta base: debe ser 0x10(a1)
# sw a0, a1                     # ❌ debe ser rs2, offset(rs1) con paréntesis
# sw a0, 8 a1                   # ❌ falta el paréntesis: 8(a1)


# ============================================================
# RESTRICCIÓN 7: Etiqueta DUPLICADA
# (dos definiciones de la misma etiqueta)
# ============================================================
# dup:
#     addi a0, x0, 1
# dup:                          # ❌ redefinición de 'dup'
#     addi a0, a0, 1

