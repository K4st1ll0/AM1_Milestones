from numpy import array, sqrt
import matplotlib.pyplot as plt
from Modules.Dynamics import N_body_problem, pack_state
from Modules.Temporal_schemes import integrate, RK4_step

# Creación de vectores 

# U = [x_1 y_1 vx_1 vy_1
#      x_2 y_2 vx_2 vy_2
#      ... ... ...  ...
#      x_n y_n vx_n vy_n]


##### Problema de 3 cuerpos iguales #####

G = 1.0

masses = array([1.0, 1.0, 1.0])

positions0 = array([
    [0.0,              1.0/sqrt(3.0)],        # cuerpo 1
    [-0.5,            -1.0/(2*sqrt(3.0))],    # cuerpo 2
    [0.5,             -1.0/(2*sqrt(3.0))]     # cuerpo 3
])

omega = sqrt(3.0)

velocities0 = array([
    [-1.0,           0.0              ],       # cuerpo 1
    [ 0.5,          -sqrt(3.0)/2      ],       # cuerpo 2
    [ 0.5,           sqrt(3.0)/2      ]        # cuerpo 3
])

U0 = pack_state(positions0, velocities0)

# Tiempo: varios periodos

T = 2.0 * 3.141592653589793 / omega    # periodo de rotación
t0 = 0.0
tf = 10 * T                            # por ejemplo, 10 periodos
N_steps = 5000                         # ajusta según precisión

F = lambda U: N_body_problem(U, masses, G=G)
sol = integrate(F, U0, t0, tf, N_steps, RK4_step)


#  Extraer posiciones

N = len(masses)
positions = sol[:, :2*N].reshape(-1, N, 2)

r1 = positions[:, 0]
r2 = positions[:, 1]
r3 = positions[:, 2]

#  Plot

plt.figure(figsize=(8,8))
plt.plot(r1[:,0], r1[:,1], label="Cuerpo 1")
plt.plot(r2[:,0], r2[:,1], label="Cuerpo 2")
plt.plot(r3[:,0], r3[:,1], label="Cuerpo 3")

# posiciones iniciales
plt.scatter(r1[0,0], r1[0,1], s=40, color='tab:blue')
plt.scatter(r2[0,0], r2[0,1], s=40, color='tab:orange')
plt.scatter(r3[0,0], r3[0,1], s=40, color='tab:green')

plt.title("Solución de 3 cuerpos iguales (órbitas circulares)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

