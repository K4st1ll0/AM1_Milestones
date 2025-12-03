import numpy as np

def cr3bp(U, mu):

    x, y, z, xdot, ydot, zdot = U

    r1 = np.sqrt( (x+mu)**2 + y**2 + z**2 )
    r2 = np.sqrt( (x-1+mu)**2 + y**2 + z**2 )

    xddot = 2*ydot + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    yddot = -2*xdot + y - (1-mu)*y/r1**3 - mu*y/r2**3
    zddot = - (1-mu)*z/r1**3 - mu*z/r2**3

    return np.array([xdot, ydot, zdot, xddot, yddot, zddot])


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

        err = np.linalg.norm(e_n1)/tol 
        h = h*min(5, max(0.1, 0.9*err**(-1/5)))

    return U5_n1


# Determinación de los puntos de Lagrange

def Jacobian(f, U, h=1e-8):
    from numpy import zeros

    
    J = zeros((len(f(U)),len(U)))

    for i in range(len(f(U))):
        for j in range(len(U)):
            
            U_plus = U.copy()
            U_plus[j] += h
            U_minus = U.copy()
            U_minus[j] -= h 
            J[i,j] = (f(U_plus)[i] - f(U_minus)[i])/(2*h)

    return J


def newton_raphson(f, U0, tol=1e-10):
    U_n = np.array(U0)

    while np.linalg.norm(f(U_n)) > tol:
        dU = np.linalg.solve(Jacobian(f, U_n), -f(U_n))
        U_n1 = U_n + dU
        U_n = U_n1

    return U_n
    

def P_Lagrange_cr3bp(U0, mu):

    f = lambda U: cr3bp(U, mu)

    U_lagrange = newton_raphson(f, U0)
    P_lagrange = U_lagrange[:3]


    return P_lagrange


#### Ejemplo puntos de Lagrange para el cr3bp ####
   
    
mu = 0.0121505856   # Tierra–Luna

# Velocidades iniciales = 0 siempre
U0_L1 = [0.8,   0.0, 0.0, 0.0, 0.0, 0.0]   # entre Tierra y Luna
U0_L2 = [1.2,   0.0, 0.0, 0.0, 0.0, 0.0]   # más allá de la Luna
U0_L3 = [-1.0,  0.0, 0.0, 0.0, 0.0, 0.0]   # al otro lado de la Tierra

U0_L4 = [0.5 - mu,  np.sqrt(3)/2,  0.0, 0.0, 0.0, 0.0]
U0_L5 = [0.5 - mu, -np.sqrt(3)/2,  0.0, 0.0, 0.0, 0.0]

P_L1 = P_Lagrange_cr3bp(U0_L1, mu)
P_L2 = P_Lagrange_cr3bp(U0_L2, mu)
P_L3 = P_Lagrange_cr3bp(U0_L3, mu)
P_L4 = P_Lagrange_cr3bp(U0_L4, mu)
P_L5 = P_Lagrange_cr3bp(U0_L5, mu)

print("Punto L1:", P_L1)
print("Punto L2:", P_L2)
print("Punto L3:", P_L3)
print("Punto L4:", P_L4)
print("Punto L5:", P_L5)

# Plot para ver los puntos de Lagrange

import matplotlib.pyplot as plt
import numpy as np

# Posiciones de los primarios en el sistema adimensional
x_earth, y_earth = -mu, 0.0
x_moon,  y_moon  = 1 - mu, 0.0

# Extraemos x, y de los puntos de Lagrange
L1_x, L1_y = P_L1[0], P_L1[1]
L2_x, L2_y = P_L2[0], P_L2[1]
L3_x, L3_y = P_L3[0], P_L3[1]
L4_x, L4_y = P_L4[0], P_L4[1]
L5_x, L5_y = P_L5[0], P_L5[1]

plt.figure(figsize=(7, 7))

# Tierra y Luna
plt.scatter(x_earth, y_earth, s=200, marker='o', label='Tierra')
plt.scatter(x_moon,  y_moon,  s=80,  marker='o', label='Luna')

# Puntos de Lagrange
plt.scatter(L1_x, L1_y, marker='x', s=80, label='L1')
plt.scatter(L2_x, L2_y, marker='x', s=80, label='L2')
plt.scatter(L3_x, L3_y, marker='x', s=80, label='L3')
plt.scatter(L4_x, L4_y, marker='x', s=80, label='L4')
plt.scatter(L5_x, L5_y, marker='x', s=80, label='L5')

# Etiquetas de texto cerca de cada punto
plt.text(L1_x, L1_y, ' L1')
plt.text(L2_x, L2_y, ' L2')
plt.text(L3_x, L3_y, ' L3')
plt.text(L4_x, L4_y, ' L4')
plt.text(L5_x, L5_y, ' L5')

plt.text(x_earth, y_earth, ' Tierra', ha='right', va='bottom')
plt.text(x_moon,  y_moon,  ' Luna',   ha='left',  va='bottom')

plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)

plt.xlabel('x (adimensional)')
plt.ylabel('y (adimensional)')
plt.title('Puntos de Lagrange en el CR3BP Tierra–Luna')
plt.gca().set_aspect('equal', 'box')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
