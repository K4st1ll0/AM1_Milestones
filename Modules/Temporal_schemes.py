from numpy import zeros, sqrt, array
from scipy import linalg

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

def Euler_step(f, U_n, delta_t):
    """
    Un solo paso de Euler explícito.

    Parámetros
    ----------
    f : función derivada F(U) del problema de Cauchy
    U_n : vector en el tiempo n (array o lista)
    delta_t : paso temporal

    Retorna
    -------
    U_n+1 : vector en el tiempo n+1 (array)
    """
    F_n = f(U_n)
    U_n1 = U_n + delta_t * F_n

    return U_n1

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

def RK4_step(f, U_n, delta_t):
    """
    Un solo paso de Runge-Kutta 4.

    Parámetros
    ----------
    f : función derivada F(U) del problema de Cauchy
    U_n : vector en el tiempo n (array o lista)
    delta_t : paso temporal

    Retorna
    -------
    U_n+1 : vector en el tiempo n+1 (array)
    """
    k1 = f(U_n)
    k2 = f(U_n + 0.5 * delta_t * k1)
    k3 = f(U_n + 0.5 * delta_t * k2)
    k4 = f(U_n + delta_t * k3)

    U_n1 = U_n + (delta_t / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return U_n1

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

def CrankNicolson_step(f, U_n, delta_t, tol = 1e-12, max_iter = 8):
    """
    Un solo paso de Crank-Nicolson.

    Parámetros
    ----------
    f : función derivada F(U) del problema de Cauchy
    U_n : vector en el tiempo n (array o lista)
    delta_t : paso temporal
    tol : tolerancia para el criterio de convergencia
    max_iter : número máximo de iteraciones para resolver la ecuación implícita

    Retorna
    -------
    U_n+1 : vector en el tiempo n+1 (array)
    """
    U_old = U_n + delta_t * f(U_n)  # Paso inicial con Euler

    for k in range(max_iter):

        U_prev = U_old.copy()

        F_n = f(U_n)
        F_old = f(U_old)

        # fórmula implícita
        U_old = U_n + 0.5 * delta_t * (F_n + F_old)

        # convergencia
        if sqrt(((U_old - U_prev)**2).sum()) < tol:
            break

    return U_old

def inverse_euler(f, U0, tf, t0 = 0, N = 10000, tol = 1e-12, max_iter = 8):
    """
    Euler inverso.

    Parámetros
    ----------
    f : función derivada F(U) del problema de Cauchy
    U0 : vector inicial (array o lista)
    delta_t : paso temporal
    N : número de pasos
    tol : tolerancia para el criterio de convergencia
    max_iter : número máximo de iteraciones para resolver la ecuación implícita

    Retorna
    -------
    U : matriz (N+1, len(U0)) con la evolución temporal
    """
    delta_t = (tf - t0) / N
    U_ie = zeros((N+1, len(U0)))
    U_ie[0, :] = U0

    for n in range(N):

        U_old = U_ie[n, :].copy()  # Paso inicial

        for k in range(max_iter):

            U_prev = U_old.copy()

            F_old = f(U_old)

            # fórmula implícita
            U_old = U_ie[n, :] + delta_t * F_old

            # convergencia
            if sqrt(((U_old - U_prev)**2).sum()) < tol:
                break

        U_ie[n+1, :] = U_old

    return U_ie

def inverse_euler_step(f, U_n, delta_t, tol = 1e-12, max_iter = 8):
    """
    Un solo paso de Euler inverso.

    Parámetros
    ----------
    f : función derivada F(U) del problema de Cauchy
    U_n : vector en el tiempo n (array o lista)
    delta_t : paso temporal
    tol : tolerancia para el criterio de convergencia
    max_iter : número máximo de iteraciones para resolver la ecuación implícita

    Retorna
    -------
    U_n+1 : vector en el tiempo n+1 (array)
    """
    U_old = U_n.copy()  # Paso inicial

    for k in range(max_iter):

        U_prev = U_old.copy()

        F_old = f(U_old)

        # fórmula implícita
        U_old = U_n + delta_t * F_old

        # convergencia
        if sqrt(((U_old - U_prev)**2).sum()) < tol:
            break

    return U_old

def integrate(f, U0, t0, tf, N, method):
    """
    Integración completa con el método seleccionado.

    Parámetros
    ----------
    f : función derivada F(U) del problema de Cauchy
    U0 : vector inicial (array o lista)
    t0 : tiempo inicial
    tf : tiempo final
    N : número de pasos
    method : método de integración ('Euler', 'RK4', 'CrankNicolson', 'inverse_euler')

    Retorna
    -------
    U : matriz (N+1, len(U0)) con la evolución temporal
    """
    dt = (tf - t0) / N

    U = zeros((N+1, len(U0)))
    U[0, :] = U0

    for n in range(N):
        U[n+1, :] = method(f, U[n, :], dt)
    
    return U


def LeapFrog_step(f, U, dt):
    x = U[0]
    v = U[1]

    a = f(U)[1]   

    v_half = v + 0.5*dt*a

    x_new = x + dt*v_half

    U_temp = array([x_new, v_half])
    a_new = f(U_temp)[1]

    v_new = v_half + 0.5*dt*a_new

    return array([x_new, v_new])


def RK45(f, U_n, tf, t0=0, N=10000, tol=1e-6):
    """
    Método Runge-Kutta-Fehlberg 45
    
    Recibe: 
        f: derivada del sistema a resolver 
        U_n: vector de variables
        tf: tiempo final
        t0: tiempo inicial
        N: número de pasos
        tol: tolerancia para el error

    Devuelve: 
        U_n5: integración del sistema en el tiempo tf
    
    """
    delta_t = (tf - t0) / N
    h = delta_t/1000 
    err = 2  # Inicialización del error para entrar al bucle

    while err > 1:

        k1 = f(U_n)
        k2 = f(U_n + h*1/4*k1)
        k3 = f(U_n + h*(3/32*k1 + 9/32*k2))
        k4 = f(U_n + h*(1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3))
        k5 = f(U_n + h*(439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4))
        k6 = f(U_n + h*(-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5))

        U4_n1 = U_n + h*(25/216*k1 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
        U5_n1 = U_n + h*(16/135*k1 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)

        e_n1 = U5_n1 - U4_n1

        err = linalg.norm(e_n1)/tol 
        h = h*min(5, max(0.1, 0.9*err**(-1/5)))

    return U5_n1