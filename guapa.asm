    li a0, 11125                # n = 5
    li t0, 0                # suma = 0
    li t1, 1                # i = 1
loop:
    blt a0, t1, end         # if (t1 > a0) goto end  (bgt t1,a0,end ≡ blt a0,t1,end)
    add t0, t0, t1          # suma += i
    addi t1, t1, 1          # i++
    beq x0, x0, loop        # salto incondicional (siempre verdadero)
end:
    ebreak                  # punto de parada (depuración)
