li a0, 5    # n = 5
li t0, 0    # sum = 0
li t1, 1    #

loop:
    bgt t1, a0, end
    add t0, t0, t1
    addi t1, t1, 1
    beq zero, zero, loop
end: