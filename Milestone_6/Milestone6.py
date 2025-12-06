import numpy as np
from Modules.Dynamics import P_Lagrange_cr3bp


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
