import numpy as np
from Modules.Dynamics import P_Lagrange_cr3bp, estability_cr3bp, cr3bp
from Modules.Math import Jacobian, analyze_eigval
import matplotlib.pyplot as plt
from Modules.Temporal_schemes import RK45, RK4_step, integrate


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


### Estabilidad de los puntos de Lagrange ###

L_points = [P_L1, P_L2, P_L3, P_L4, P_L5]

estability_cr3bp(mu, L_points)

### Órbitas alrededor de los puntos de Lagrange ###
# Doy una pequeña perturbación epsilon a las posiciones de los puntos de Lagrange
# y calculo las órbitas resultantes

epsilon = 1e-3

U0_L1_perturbed = np.array([P_L1[0] + epsilon, P_L1[1], P_L1[2], 0.0, 0.0, 0.0])
U0_L2_perturbed = np.array([P_L2[0] + epsilon, P_L2[1], P_L2[2], 0.0, 0.0, 0.0])
U0_L3_perturbed = np.array([P_L3[0] + epsilon, P_L3[1], P_L3[2], 0.0, 0.0, 0.0])
U0_L4_perturbed = np.array([P_L4[0] + epsilon, P_L4[1], P_L4[2], 0.0, 0.0, 0.0])
U0_L5_perturbed = np.array([P_L5[0] + epsilon, P_L5[1], P_L5[2], 0.0, 0.0, 0.0])

# RK45

U_L1_perturbed = RK45(lambda U: cr3bp(U, mu), U0_L1_perturbed, tf=2, t0=0, N=1000)
U_L2_perturbed = RK45(lambda U: cr3bp(U, mu), U0_L2_perturbed, tf=2, t0=0, N=1000)
U_L3_perturbed = RK45(lambda U: cr3bp(U, mu), U0_L3_perturbed, tf=2, t0=0, N=1000)
U_L4_perturbed = RK45(lambda U: cr3bp(U, mu), U0_L4_perturbed, tf=2, t0=0, N=1000)
U_L5_perturbed = RK45(lambda U: cr3bp(U, mu), U0_L5_perturbed, tf=2, t0=0, N=1000)
### Representación de las órbitas ###

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))

# Órbitas alrededor de los Lagrange
plt.plot(U_L1_perturbed[:,0], U_L1_perturbed[:,1], label="Órbita L1")
plt.plot(U_L2_perturbed[:,0], U_L2_perturbed[:,1], label="Órbita L2")
plt.plot(U_L3_perturbed[:,0], U_L3_perturbed[:,1], label="Órbita L3")
plt.plot(U_L4_perturbed[:,0], U_L4_perturbed[:,1], label="Órbita L4")
plt.plot(U_L5_perturbed[:,0], U_L5_perturbed[:,1], label="Órbita L5")

# Añadimos Tierra y Luna
plt.scatter(-mu, 0, s=120, color='blue', label='Tierra')
plt.scatter(1-mu, 0, s=60, color='gray', label='Luna')

plt.xlabel("x")
plt.ylabel("y")
plt.title("Órbitas alrededor de los puntos de Lagrange (CR3BP)")
plt.gca().set_aspect("equal", "box")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# RK4_step

U_L1_perturbed_rk4 = integrate(lambda U: cr3bp(U, mu), U0_L1_perturbed, t0=0, tf=30,  N=1000, method=RK4_step)
U_L2_perturbed_rk4 = integrate(lambda U: cr3bp(U, mu), U0_L2_perturbed, t0=0, tf=30,  N=1000, method=RK4_step)
U_L3_perturbed_rk4 = integrate(lambda U: cr3bp(U, mu), U0_L3_perturbed, t0=0, tf=30,  N=1000, method=RK4_step)
U_L4_perturbed_rk4 = integrate(lambda U: cr3bp(U, mu), U0_L4_perturbed, t0=0, tf=30,  N=1000, method=RK4_step)
U_L5_perturbed_rk4 = integrate(lambda U: cr3bp(U, mu), U0_L5_perturbed, t0=0, tf=30,  N=1000, method=RK4_step)
### Representación de las órbitas ###

plt.figure(figsize=(8, 8))
# Órbitas alrededor de los Lagrange
plt.plot(U_L1_perturbed_rk4[:,0], U_L1_perturbed_rk4[:,1], label="Órbita L1 RK4")
plt.plot(U_L2_perturbed_rk4[:,0], U_L2_perturbed_rk4[:,1], label="Órbita L2 RK4")
plt.plot(U_L3_perturbed_rk4[:,0], U_L3_perturbed_rk4[:,1], label="Órbita L3 RK4")
plt.plot(U_L4_perturbed_rk4[:,0], U_L4_perturbed_rk4[:,1], label="Órbita L4 RK4")
plt.plot(U_L5_perturbed_rk4[:,0], U_L5_perturbed_rk4[:,1], label="Órbita L5 RK4")   
# Añadimos Tierra y Luna
plt.scatter(-mu, 0, s=120, color='blue', label='Tierra')
plt.scatter(1-mu, 0, s=60, color='gray', label='Luna')  

plt.xlabel("x")
plt.ylabel("y") 
plt.title("Órbitas alrededor de los puntos de Lagrange (CR3BP) con RK4")
plt.gca().set_aspect("equal", "box")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()








    
