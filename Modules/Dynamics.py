from numpy import array, zeros_like, dot, hstack, sqrt
import numpy as np
from Modules.Math import newton_raphson, Jacobian, analyze_eigval


def unpack_state(U, N):
    """
    Recibe:
        U: vector de estado de tamaño 4N
        N: número de cuerpos
    Devuelve:
        positions: array de shape (N, 2) con [x_i, y_i]
        velocities: array de shape (N, 2) con [vx_i, vy_i]
    """
    
    positions = U[:2*N].reshape(N, 2)  # Primeras 2N posiciones
    velocities = U[2*N:].reshape(N, 2) # Últimas 2N velocidades

    return positions, velocities


def pack_state(positions, velocities):
    """
    Hace lo contrario que unpack_state, pasa de (N,2) + (N,2) a un vector de longitud 4N.
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

def cr3bp(U, mu):

    x, y, z, xdot, ydot, zdot = U

    r1 = sqrt( (x+mu)**2 + y**2 + z**2 )
    r2 = sqrt( (x-1+mu)**2 + y**2 + z**2 )

    xddot = 2*ydot + x - (1-mu)*(x+mu)/r1**3 - mu*(x-1+mu)/r2**3
    yddot = -2*xdot + y - (1-mu)*y/r1**3 - mu*y/r2**3
    zddot = - (1-mu)*z/r1**3 - mu*z/r2**3

    return array([xdot, ydot, zdot, xddot, yddot, zddot])


def P_Lagrange_cr3bp(U0, mu):

    f = lambda U: cr3bp(U, mu)

    U_lagrange = newton_raphson(f, U0)
    P_lagrange = U_lagrange[:3]


    return P_lagrange


def estability_cr3bp(mu, L_points):

    P_L1, P_L2, P_L3, P_L4, P_L5 = L_points

    U_L1 = np.zeros(6)
    U_L2 = np.zeros(6)
    U_L3 = np.zeros(6)
    U_L4 = np.zeros(6)
    U_L5 = np.zeros(6)

    U_L1[:3] = P_L1
    U_L2[:3] = P_L2
    U_L3[:3] = P_L3
    U_L4[:3] = P_L4
    U_L5[:3] = P_L5

    J_L1 = Jacobian(lambda U: cr3bp(U, mu), U_L1)
    J_L2 = Jacobian(lambda U: cr3bp(U, mu), U_L2)
    J_L3 = Jacobian(lambda U: cr3bp(U, mu), U_L3)
    J_L4 = Jacobian(lambda U: cr3bp(U, mu), U_L4)
    J_L5 = Jacobian(lambda U: cr3bp(U, mu), U_L5)

    eigval_L1 = np.linalg.eigvals(J_L1)
    eigval_L2 = np.linalg.eigvals(J_L2)
    eigval_L3 = np.linalg.eigvals(J_L3)
    eigval_L4 = np.linalg.eigvals(J_L4)
    eigval_L5 = np.linalg.eigvals(J_L5)

    E_L1 = analyze_eigval(eigval_L1)
    E_L2 = analyze_eigval(eigval_L2)
    E_L3 = analyze_eigval(eigval_L3)
    E_L4 = analyze_eigval(eigval_L4)
    E_L5 = analyze_eigval(eigval_L5)

    print("Estabilidad L1:", E_L1)
    print("Estabilidad L2:", E_L2)
    print("Estabilidad L3:", E_L3)
    print("Estabilidad L4:", E_L4)
    print("Estabilidad L5:", E_L5)


