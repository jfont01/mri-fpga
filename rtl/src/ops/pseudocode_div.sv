START

S = 0
Q = N
count = 0
NB_QUOTIENT = NB_NUM

loop:
    {S, Q} = {S, Q} << 1

    sub = S - D

    if sub >= 0:
        S = sub
        Q[0] = 1
    else:
        S = S
        Q[0] = 0

    count++

    if count == NB_QUOTIENT:
        Q = quotient
        S = remainder
        END
    else:
        loop






START

sign_N = N[MSB]
sign_D = D[MSB]
sign_Q = sign_N XOR sign_D

SHIFT = NBF_QUOTIENT_OUT + NBF_DEN - NBF_NUM
NB_QUOTIENT_INTERNAL = NB_NUM + SHIFT

N_scaled = abs(N) << SHIFT
S = 0
Q = N_scaled
count = 0

loop:
    {S, Q} = {S, Q} << 1

    sub = S - abs(D)

    if sub >= 0:
        S = sub
        Q[0] = 1
    else:
        S = S
        Q[0] = 0

    count++

    if count == NB_QUOTIENT_INTERNAL:
        if (S << 1) >= abs(D):      // ¿S / abs(D) >= 0.5?
            need_round_flag = 1
        else:
            need_round_flag = 0

        if sign_Q:
            quotient_out = -Q
        else:
            quotient_out = Q


        END

    else:
        loop
