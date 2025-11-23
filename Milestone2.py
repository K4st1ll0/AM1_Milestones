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
    

###########################################################################
#########################   Test funciones   ##############################
###########################################################################

#### Variables iniciales ####

x0 = 1
y0 = 0
u0 = 0 
v0 = 1

N = 10000

t0 = 0
tf = 20

U0 = array([x0, y0, u0, v0])

# Prueba función Euler

U_Euler = Euler(f, U0, tf, t0, N)

plt.figure(figsize=(7,4))
plt.plot(U_Euler[:,0], U_Euler[:,1])
plt.axis('equal')  
plt.title('Euler Explícito')
plt.show()

# Prueba función RK4

U_RK4 = RK4(f, U0, tf, t0, N)

plt.figure(figsize=(7,4))
plt.plot(U_RK4[:,0], U_RK4[:,1])
plt.axis('equal')  
plt.title('RUNGE-KUTTA 4')
plt.show()

# Prueba función Crank-Nicolson

U_cn = CrankNicolson(f, U0, tf, t0, N)

plt.figure(figsize=(7,4))
plt.plot(U_cn[:,0], U_cn[:,1])
plt.axis('equal')  
plt.title('Crank-Nicolson')
plt.show()

# Prueba función Inverse Euler

U_ie = inverse_euler(f, U0, tf, t0, N)

plt.figure(figsize=(7,4))
plt.plot(U_ie[:,0], U_ie[:,1]) 
plt.axis('equal')  
plt.title('Inverse Euler')
plt.show()

    
# Prueba integrate

U_int = integrate(f, U0, t0, tf, N, method=CrankNicolson_step) # Cambiar método aquí

plt.figure(figsize=(7,4))
plt.plot(U_int[:,0], U_int[:,1])  
plt.axis('equal')  
plt.title('Integrate ')
plt.show()