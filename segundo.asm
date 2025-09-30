.text
start:
    # U-type
    lui   t0, 25889988
    auipc t1, 0x1

    # I-type (arith/shift/load)
    addi  s0, x0, -16
    slli  s1, s0, 3
    addi  t2, x0, 0
    lw    a0, 0(t2)

    # R-type
    add   a1, a0, s1
    and   a2, a1, s0

    # S-type
    sw    a2, 4(t2)

    # B-type
    beq   a2, a2, after

    addi  a3, x0, 999   # (salteado si se toma el branch)

after:
    # J-type
    jal   ra, finish

    addi  a4, x0, 123   # (no debería ejecutarse por el jal)

finish:
    ret
