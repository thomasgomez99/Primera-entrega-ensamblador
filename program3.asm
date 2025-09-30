.text
start:
    li   a0, 1000      # pseudo → addi o lui+addi
    addi a1, a0, -5
    beq  a1, x0, done
    j    start         # pseudo → jal x0, start
done:
    ret                # pseudo → jalr x0, ra, 0
