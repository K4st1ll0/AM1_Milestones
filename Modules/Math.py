from numpy import array, abs, linspace, meshgrid
from matplotlib import pyplot as plt
from scipy import linalg

def f(U):
    ''' Retorna el vector de derivadas '''
    x, y, u, v = U
    dudt = -x / ((x**2 + y**2)**(3/2))
    dvdt = -y / ((x**2 + y**2)**(3/2))

    return array([u, v, dudt, dvdt])


def F(U):
    x = U[0]
    v = U[1]
    return array([v, -x])


def plot_region(scheme, title, limit=1, x = linspace(-3, 3, 800), y = linspace(-3, 3, 800)):

    """
    Grafica la región de estabilidad de un esquema temporal dado.
    Parámetros:
    - scheme: Nombre del esquema temporal ("Euler", "inverse_euler", "RK4", "CrankNicolson", "LeapFrog").
    - title: Título del gráfico.
    - limit: Límite para la región estable (por defecto es 1).
    - x, y: Rango de valores en el plano complejo (por defecto de -3 a 3).
    Retorna:
    - Gráfico de la región de estabilidad.
    
    """

    # Malla en el plano complejo
    
    X, Y = meshgrid(x, y)
    Z = X + 1j*Y

    if scheme == "Euler":
        R = 1 + Z
    elif scheme == "inverse_euler":
        R = 1 / (1 - Z)
    elif scheme == "RK4":
        R = 1 + Z + (Z**2)/2 + (Z**3)/6 + (Z**4)/24
    elif scheme == "CrankNicolson":
        R = (1 + Z/2) / (1 - Z/2)
    elif scheme == "LeapFrog":
        R = Z


    plt.figure(figsize=(6,6))
    
    mask = abs(R) <= limit

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


def Jacobian(f, U, h=1e-8):
    from numpy import zeros

    
    J = zeros((len(f(U)),len(U)))

    for i in range(len(f(U))):
        for j in range(len(U)):
            
            U_plus = U.copy()
            U_plus[j] += h
            U_minus = U.copy()
            U_minus[j] -= h 
            J[i,j] = (f(U_plus)[i] - f(U_minus)[i])/(2*h)

    return J


def newton_raphson(f, U0, tol=1e-10):
    U_n = array(U0)

    while linalg.norm(f(U_n)) > tol:
        dU = linalg.solve(Jacobian(f, U_n), -f(U_n))
        U_n1 = U_n + dU
        U_n = U_n1

    return U_n