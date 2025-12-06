import numpy as np
import matplotlib.pyplot as plt
from Modules.Errors import scheme_numerical_error, convergence_rate
from Modules.Temporal_schemes import Euler, RK4, CrankNicolson, inverse_euler
from Modules.Math import f






x0 = 1
y0 = 0
u0 = 0 
v0 = 1

N = 10000

t0 = 0
tf = 20

U0 = np.array([x0, y0, u0, v0])


Eh_eu, UR_eu, E_global_eu = scheme_numerical_error(f, U0, t0, tf, N, Euler, 1)
Eh_rk, UR_rk, E_global_rk = scheme_numerical_error(f, U0, t0, tf, N, RK4, 4)
Eh_cn, UR_cn, E_global_cn = scheme_numerical_error(f, U0, t0, tf, N, CrankNicolson, 2)
Eh_ie, UR_ie, E_global_ie = scheme_numerical_error(f, U0, t0, tf, N, inverse_euler, 1)

t = np.linspace(t0, tf, N + 1)

# Error norms over time (Euclidean norm of Eh)
plt.figure(figsize=(8, 4))
plt.plot(t, np.linalg.norm(Eh_eu, axis=1), label='Euler (p=1)')
plt.plot(t, np.linalg.norm(Eh_rk, axis=1), label='RK4 (p=4)')
plt.plot(t, np.linalg.norm(Eh_cn, axis=1), label='Crank-Nicolson (p=2)')
plt.plot(t, np.linalg.norm(Eh_ie, axis=1), label='Inverse Euler (p=1)')
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('||E_h(t)||_2')
plt.title('Norma del error estimado por extrapolación de Richardson')
plt.legend()
plt.tight_layout()

# Phase-space trajectories from Richardson solutions (x vs y)
plt.figure(figsize=(6, 6))
plt.plot(UR_eu[:, 0], UR_eu[:, 1], label='Euler Richardson')
plt.plot(UR_rk[:, 0], UR_rk[:, 1], label='RK4 Richardson')
plt.plot(UR_cn[:, 0], UR_cn[:, 1], label='Crank-Nicolson Richardson')
plt.plot(UR_ie[:, 0], UR_ie[:, 1], label='Inverse Euler Richardson')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trayectorias (soluciones Richardson)')
plt.axis('equal')
plt.legend()
plt.tight_layout()

plt.show()

Eh_eu, UR_eu, E_global_eu = scheme_numerical_error(f, U0, t0, tf, N, Euler, 1)
Eh_rk, UR_rk, E_global_rk = scheme_numerical_error(f, U0, t0, tf, N, RK4, 4)
Eh_cn, UR_cn, E_global_cn = scheme_numerical_error(f, U0, t0, tf, N, CrankNicolson, 2)
Eh_ie, UR_ie, E_global_ie = scheme_numerical_error(f, U0, t0, tf, N, inverse_euler, 1)

Eh_eu2, UR_eu2, E_global_eu2 = scheme_numerical_error(f, U0, t0, tf, 2*N, Euler, 1)
Eh_rk2, UR_rk2, E_global_rk2 = scheme_numerical_error(f, U0, t0, tf, 2*N, RK4, 4)
Eh_cn2, UR_cn2, E_global_cn2 = scheme_numerical_error(f, U0, t0, tf, 2*N, CrankNicolson, 2)
Eh_ie2, UR_ie2, E_global_ie2 = scheme_numerical_error(f, U0, t0, tf, 2*N, inverse_euler, 1)

rate_eu = convergence_rate(E_global_eu, E_global_eu2)
rate_rk = convergence_rate(E_global_rk, E_global_rk2)
rate_cn = convergence_rate(E_global_cn, E_global_cn2)
rate_ie = convergence_rate(E_global_ie, E_global_ie2)

print(f"Convergence rate Euler: {rate_eu}")
print(f"Convergence rate RK4: {rate_rk}")   
print(f"Convergence rate Crank-Nicolson: {rate_cn}")
print(f"Convergence rate Inverse Euler: {rate_ie}")


N_values = np.array([200, 1600, 3200, 6400, 12800, 25600])  

errors_eu = []
errors_rk = []
errors_cn = []
errors_ie = []

# ==== CALCULAR ERRORES GLOBALES PARA CADA N ====
for N in N_values:
    _, _, e_eu = scheme_numerical_error(f, U0, t0, tf, N, Euler, 1)
    _, _, e_rk = scheme_numerical_error(f, U0, t0, tf, N, RK4, 4)
    _, _, e_cn = scheme_numerical_error(f, U0, t0, tf, N, CrankNicolson, 2)
    _, _, e_ie = scheme_numerical_error(f, U0, t0, tf, N, inverse_euler, 1)

    errors_eu.append(e_eu)
    errors_rk.append(e_rk)
    errors_cn.append(e_cn)
    errors_ie.append(e_ie)

# ==== GRAFICAR log(||E||) vs N ====
plt.figure(figsize=(8,5))
plt.loglog(N_values, errors_eu, 'o-', label="Euler (p=1)")
plt.loglog(N_values, errors_rk, 'o-', label="RK4 (p=4)")
plt.loglog(N_values, errors_cn, 'o-', label="Crank–Nicolson (p=2)")
plt.loglog(N_values, errors_ie, 'o-', label="Inverse Euler (p=1)")

plt.xlabel("log(N)")
plt.ylabel("||E_h||")
plt.title("Convergencia: log(||E_h||) vs log(N)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

