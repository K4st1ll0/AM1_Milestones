from numpy import array
from Modules.Temporal_schemes import Euler_step, inverse_euler_step, RK4_step, CrankNicolson_step, LeapFrog_step, integrate
from Modules.Math import F, plot_region
import matplotlib.pyplot as plt
import numpy as np


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

plot_region("Euler",  "Región de estabilidad — Euler explícito")
plot_region("inverse_euler", "Región de estabilidad — Euler inverso")
plot_region("CrankNicolson",  "Región de estabilidad — Crank-Nicolson")
plot_region("RK4",  "Región de estabilidad — RK4")
plot_region("LeapFrog",  "Región de estabilidad — Leap-Frog", limit=2)

