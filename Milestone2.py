from numpy import array, zeros, sqrt
from matplotlib import pyplot as plt

def f(U):
    ''' Retorna el vector de derivadas '''
    x, y, u, v = U
    dudt = -x / ((x**2 + y**2)**(3/2))
    dvdt = -y / ((x**2 + y**2)**(3/2))

    return array([u, v, dudt, dvdt])

def Euler(f, U0, tf, t0 = 0, N = 10000):
    """
    Integración completa con Euler explícito.

    Parámetros
    ----------
    f : función derivada F(U) del problema de Cauchy
    U0 : vector inicial (array o lista)
    delta_t : paso temporal
    N : número de pasos

    Retorna
    -------
    U : matriz (N+1, len(U0)) con la evolución temporal
    """
    
    delta_t = (tf - t0) / N
    U = zeros((N+1, len(U0)))
    F = zeros((N+1, len(U0)))

    U[0, :] = U0

    for n in range(N):

        F[n, :] = f(U[n, :])

        U[n+1, :] = U[n, :] + delta_t * F[n, :]

    return U

def RK4(f, U0, tf, t0 = 0, N = 10000):
    """
    Runge-Kutta 4.

    Parámetros
    ----------
    f : función derivada F(U) del problema de Cauchy
    U0 : vector inicial (array o lista)
    delta_t : paso temporal
    N : número de pasos

    Retorna
    -------
    U : matriz (N+1, len(U0)) con la evolución temporal
    """
    delta_t = (tf - t0) / N
    U = zeros((N+1, len(U0)))

    U[0, :] = U0

    for n in range(N):

        k1 = f(U[n, :])
        k2 = f(U[n, :] + 0.5 * delta_t * k1)
        k3 = f(U[n, :] + 0.5 * delta_t * k2)
        k4 = f(U[n, :] + delta_t * k3)

        U[n+1, :] = U[n, :] + (delta_t / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return U

def CrankNicolson(f, U0, tf, t0 = 0, N = 10000, max_iter = 8, tol = 1e-12):
    """
    Crank-Nicolson.

    Parámetros
    ----------
    f : función derivada F(U) del problema de Cauchy
    U0 : vector inicial (array o lista)
    delta_t : paso temporal
    N : número de pasos
    max_iter : número máximo de iteraciones para resolver la ecuación implícita
    tol : tolerancia para el criterio de convergencia

    Retorna
    -------
    U : matriz (N+1, len(U0)) con la evolución temporal
    """
    delta_t = (tf - t0) / N
    U_cn = zeros((N+1, len(U0)))
    U_cn[0, :] = U0

    for n in range(N):

        U_old = U_cn[n, :] + delta_t * f(U_cn[n, :])  # Paso inicial con Euler

        for k in range(max_iter):

            U_prev = U_old.copy()

            F_n = f(U_cn[n, :])
            F_old = f(U_old)

            # fórmula implícita
            U_old = U_cn[n, :] + 0.5 * delta_t * (F_n + F_old)

            # convergencia
            if sqrt(((U_old - U_prev)**2).sum()) < tol:
                break

        U_cn[n+1, :] = U_old

    return U_cn