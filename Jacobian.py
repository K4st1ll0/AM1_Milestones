from numpy import zeros

def Jacobian(F, x0, h=1e-5):
    N = len(x0)

    J = zeros((N,N))
    delta_x = zeros(N)

    for i in range(0,N):
        delta_x[:] = 0
        delta_x[i] = h
        J[:,i] = (F(x0+delta_x) - F(x0-delta_x)) / (2*h)
       
    return J