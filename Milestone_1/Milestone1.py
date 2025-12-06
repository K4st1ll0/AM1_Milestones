from numpy import array, zeros, sqrt
from matplotlib import pyplot as plt

#### Funciones ####

def f(U):
    ''' Retorna el vector de derivadas '''
    x, y, u, v = U
    dudt = -x / ((x**2 + y**2)**(3/2))
    dvdt = -y / ((x**2 + y**2)**(3/2))

    return array([u, v, dudt, dvdt])


#### Variables iniciales ####

x0 = 1
y0 = 0
u0 = 0 
v0 = 1

N = 10000

t0 = 0
tf = 20
delta_t = (tf - t0) / N

U0 = array([x0, y0, u0, v0])


###############
#### EULER ####
###############

U = zeros([N+1, len(U0)])
F = zeros([N+1, len(U0)])

U[0,:] = U0

for N in range (0,N):

    F[N,:] = f(U[N,:])

    U[N+1,:] = U[N,:] + delta_t * F[N,:]

U_Euler = zeros([N+1, len(U0)])
U_Euler = U

# GRAFICA EULER #

plt.figure(figsize=(7,4))
plt.plot(U_Euler[:,0], U_Euler[:,1])
plt.axis('equal')  
plt.title('Euler Explícito')
plt.show()


########################
#### RUNGE KUTTA 4  ####
########################

t = zeros(N+1)
t[0] = t0
U = zeros([N+1, len(U0)])
U[0,:] = U0
for n in range(N):
    
    k1 = f(U[n])
    k2 = f(U[n] + 0.5*delta_t*k1)
    k3 = f(U[n] + 0.5*delta_t*k2)
    k4 = f(U[n] + delta_t*k3)

    U[n+1] = U[n] + (delta_t/6)*(k1 + 2*k2 + 2*k3 + k4)
    t[n+1] = t[n] + delta_t

U_RK4 = zeros([N+1, len(U0)])
U_RK4 = U

# GRAFICA RK4 #

plt.figure(figsize=(7,4))
plt.plot(U_RK4[:,0], U_RK4[:,1])
plt.axis('equal')  
plt.title('RUNGE-KUTTA 4')
plt.show()


########################
#### Crank-Nicolson ####
########################

U_cn = zeros([N+1, len(U0)])
U_cn[0,:] = U0

max_iter = 8
tol = 1e-12


for n in range(N):

    U_old = U_cn[n,:] + delta_t * f(U_cn[n,:]) # Paso inicial con Euler 

    # iteraciones implícitas
    for k in range(max_iter):

        U_prev = U_old.copy()

        F_n = f(U_cn[n,:])
        F_old = f(U_old)

        # fórmula implícita
        U_old = U_cn[n,:] + 0.5*delta_t*(F_n + F_old)

        # convergencia
        if sqrt(((U_old - U_prev)**2).sum()) < tol:
            break

    U_cn[n+1,:] = U_old
 

# GRAFICA CN #

plt.figure(figsize=(7,4))
plt.plot(U_cn[:,0], U_cn[:,1])
plt.axis('equal')  
plt.title('CRANK-NICOLSON')
plt.show()



### Gráfica comparativa ####

fig, ax = plt.subplots(figsize=(7,4))

ax.plot(U_Euler[:,0], U_Euler[:,1], label='Euler Explícito')
ax.plot(U_RK4[:,0],   U_RK4[:,1],   label='Runge–Kutta 4')
ax.plot(U_cn[:,0],    U_cn[:,1],    label='Crank–Nicolson')   

ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_title('Comparación: Euler vs RK4 vs Crank–Nicolson')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.legend()

fig.tight_layout()
plt.show()






