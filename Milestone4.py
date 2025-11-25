from numpy import array
from Milestone2 import *
import matplotlib.pyplot as plt

def F(U):
    x = U[0]
    v = U[1]
    return array([v, -x])

import numpy as np

def LeapFrog_step(f, U, dt):
    x = U[0]
    v = U[1]

    a = f(U)[1]   

    v_half = v + 0.5*dt*a

    x_new = x + dt*v_half

    U_temp = np.array([x_new, v_half])
    a_new = f(U_temp)[1]

    v_new = v_half + 0.5*dt*a_new

    return np.array([x_new, v_new])


# Condiciones iniciales
U0 = array([1, 0])
t0 = 0
tf = 10
N = 1000

oscilador_eu = integrate(F, U0, t0, tf, N, method=Euler_step)
oscilador_ieu = integrate(F, U0, t0, tf, N, method=inverse_euler_step)
oscilador_rk = integrate(F, U0, t0, tf, N, method=RK4_step)
oscilador_cn = integrate(F, U0, t0, tf, N, method=CrankNicolson_step)
oscilador_lf = integrate(F, U0, t0, tf, N, method=LeapFrog_step)

# Regiones de estabilidad

# Malla en el plano complejo
x = np.linspace(-3, 3, 800)
y = np.linspace(-3, 3, 800)
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y


R_eu = 1 + Z
R_ieu = 1 / (1 - Z)
R_rk = 1 + Z + (Z**2)/2 + (Z**3)/6 + (Z**4)/24
R_cn = (1 + Z/2) / (1 - Z/2)
R_lf = Z

def plot_region(R, title, limit=1):
    plt.figure(figsize=(6,6))
    
    mask = np.abs(R) <= limit

    # Pintamos la región estable en azul claro
    plt.imshow(mask.astype(int),
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower',
               cmap='Blues',
               alpha=0.8)

    # Ejes
    plt.axhline(0, color='k', linewidth=1)
    plt.axvline(0, color='k', linewidth=1)

    plt.title(title)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.gca().set_aspect("equal")
    plt.show()


plot_region(R_eu,  "Región de estabilidad — Euler explícito")
plot_region(R_ieu, "Región de estabilidad — Euler inverso")
plot_region(R_cn,  "Región de estabilidad — Crank–Nicolson")
plot_region(R_rk,  "Región de estabilidad — RK4")
plot_region(R_lf,  "Región de estabilidad — Leap–Frog", limit=2)

