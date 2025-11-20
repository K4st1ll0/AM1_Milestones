
def Cauchy_Error(F, U0, t, temporal_scheme, q):

    E = zeros((N+1, Nv))
    t1 = t[:]
    t2 = linspace(t[0], t[N], 2*N+1)

    U1 = Cauchy_problem(F, U0, t1, temporal_scheme)
    U2 = Cauchy_problem(F, U0, t2, temporal_scheme)

    for n in range(0,N+1): 
        E[n,:] = (U2[2*n,:]-U1[n,:])/ (1-1/2**q)
    return U1, E

def comvergence_rate(temporal_scheme, F = oscilador, U0=U0, t=t)
    N_meshes  = 5
    N_mesh = array([10, 20, 40, 60, 80])
    E = zeros(N_meshes)
    logN = log(N_mesh)
    logE = zeros(N_meshes)
    N = len(t)
    for n in range(N_meshes):
        t_n = linspace(t[0], t[N], N_mesh[n])
        U1, E[n] = Cauchy_Error(F, U0, t_n, temporal_scheme, q)
    
    logN = log(N_mesh)
    logE = log(E)


