from numpy import array
from Modules.Temporal_schemes import Euler, RK4, CrankNicolson, inverse_euler, CrankNicolson_step, integrate
from Modules.Math import f
from matplotlib import pyplot as plt


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
    
