.data
inputarray0: .word 64
inputarray1: .word 25
inputarray2: .word 12
inputarray3: .word 22
inputarray4: .word 11
inputarray5: .word 90
inputarray6: .word 34
inputarray7: .word 78

.text
lui a5, %hi(inputarray1)
addi a5, a5, %lo(inputarray1)  

lw a0, 0(a5)        # Load inputarray[0] into a0
lw a1, 4(a5)        # Load inputarray[1] into a1
lw a2, 8(a5)        # Load inputarray[2] into a2
lw a3, 12(a5)       # Load inputarray[3] into a3
lw t0, 16(a5)       # Load inputarray[4] into t0
lw t1, 20(a5)       # Load inputarray[5] into t1
lw t2, 24(a5)       # Load inputarray[6] into t2
lw t3, 28(a5)       # Load inputarray[7] into t3
