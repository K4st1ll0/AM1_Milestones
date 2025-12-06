import numpy as np

def richardson_extrapolation(U_h, U_h2, p):

    Eh = (U_h2[::2] - U_h) / (2**p - 1)
    U_Richardson = U_h2[::2] + Eh

    return Eh, U_Richardson

def global_error(E_vec):
    """
    A partir del error nodal E_vec(t) ∈ R^{N+1,4},
    devuelve el error global E_h = max_t ||E(t)||_2
    """
    err_t = np.linalg.norm(E_vec, axis=1)  
    return np.max(err_t)

def scheme_numerical_error(f, U0, t0, tf, N, method, p):

    Uh = method(f, U0, t0, tf, N)
    Uh2 = method(f, U0, t0, tf, 2*N)

    Eh, U_Richardson = richardson_extrapolation(Uh, Uh2, p)

    E_global = global_error(Eh)

    return Eh, U_Richardson, E_global

def convergence_rate(Eh1, Eh2):

    return np.log(Eh1 / Eh2) / np.log(2)