from numpy import array, diff, hstack, zeros_like, dot , sqrt, zeros
from Milestone2 import integrate, RK4_step
from matplotlib import pyplot as plt

# Creación de vectores 

# U = [x_1 y_1 vx_1 vy_2
#      x_2 y_2 vx_2 vy_2
#      ... ... ...  ...
#      x_n y_n vx_n vy_n]

def unpack_state(U, N):
    """
    Recibe:
        U: vector de estado de tamaño 4N
        N: número de cuerpos
    Devuelve:
        positions: array de shape (N, 2) con [x_i, y_i]
        velocities: array de shape (N, 2) con [vx_i, vy_i]
    """
    
    positions = U[:2*N].reshape(N, 2)  # Primeras 2N: posiciones
    velocities = U[2*N:].reshape(N, 2) # Últimas 2N: velocidades

    return positions, velocities


def pack_state(positions, velocities):
    """
    Hace lo contrario, pasa de (N,2) + (N,2) a un vector 1D de longitud 4N.
    """
    return hstack((positions.reshape(-1), velocities.reshape(-1)))


def N_body_problem(U, masas, G=1, eps=1e-9):
    N = len(masas)
    positions, velocities = unpack_state(U, N)
    accelerations = zeros_like(positions)

    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = positions[j] - positions[i]  
                dist_sq = dot(r_ij, r_ij)        
                dist = sqrt(dist_sq)           
                dist3 = dist_sq*dist + eps**3       

                accelerations[i] += G * masas[j] * r_ij / dist3

    return pack_state(velocities, accelerations)




# ######## Modelo Sol-Tierra-Luna ########

# # Unidades y constantes

# G = 2.95912208286e-4  # AU^3 / (day^2 * solar_mass)

# # Masas
# M_sun   = 1.0
# M_earth = 3.003e-6
# M_moon  = 3.694e-8

# masses = array([M_sun, M_earth, M_moon])


# # Posiciones iniciales (AU)

# positions0 = array([
#     [0.0,     0.0],            # Sol
#     [1.0,     0.0],            # Tierra
#     [1.00257, 0.0]             # Luna
# ])


# # Velocidades iniciales (AU/día)

# v_earth = array([0.0, 0.0172])     # Tierra alrededor del Sol
# v_moon_rel = array([0.0, 0.0027])  # Luna respecto a la Tierra

# velocities0 = array([
#     [0.0,     0.0],             # Sol
#     v_earth,                    # Tierra
#     v_earth + v_moon_rel        # Luna
# ])

# U0 = pack_state(positions0, velocities0)


# # Trayectorias con RK4

# t0 = 0.0
# tf = 365.0       # 1 año
# N_steps = 200000

# F = lambda U: N_body_problem(U, masses)
# N_body_solution = integrate(F, U0, t0, tf, N_steps, RK4_step)

# # Extraer las posiciones del resultado

# N = len(masses)
# positions = N_body_solution[:, :2*N]    # solo las posiciones
# positions = positions.reshape(-1, N, 2) # (tiempo, cuerpo, coords)

# sol = positions[:, 0]   # Sol
# earth = positions[:, 1] # Tierra
# moon = positions[:, 2]  # Luna


# # Plot de las órbitas

# plt.figure(figsize=(8,8))

# plt.plot(sol[:,0], sol[:,1], 'yo', label='Sol', markersize=8)               # Sol
# plt.plot(earth[:,0], earth[:,1], 'b', label='Tierra')                      # Tierra
# plt.plot(moon[:,0], moon[:,1], 'gray', label='Luna', linewidth=0.8)        # Luna

# plt.scatter(sol[0,0], sol[0,1], color='orange', s=80)  # posición inicial Sol
# plt.scatter(earth[0,0], earth[0,1], color='blue', s=40) # posición inicial Tierra
# plt.scatter(moon[0,0], moon[0,1], color='black', s=20) # posición inicial Luna

# plt.xlabel("x (UA)")
# plt.ylabel("y (UA)")
# plt.title("Órbitas Sol–Tierra–Luna (integración RK4)")
# plt.grid(True)
# plt.axis('equal')
# plt.legend()
# plt.show()

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

plt.title("Solución de Lagrange – 3 cuerpos iguales (órbitas circulares)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

