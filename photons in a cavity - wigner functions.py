import numpy as np
from qutip import Qobj, wigner
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tabulate import tabulate 
import matplotlib.animation as animation
from math import factorial

#SImulation of the time evolution of wigner functions of a coherent state for photons inside a cavity

def generate_random_density_matrix(dim):
    Z = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    H = Z + Z.conj().T #making hermitian
    D = np.dot(H, H.conj().T) #making positive semi-definite
    D = D / np.trace(D) # making trace =1
    return D

def print_matrix(matrix): #prints the density matrix in a nice way
    str_matrix = [[f'{z.real:.4f} + {z.imag:.4f}j' for z in row] for row in matrix]
    print(tabulate(str_matrix, tablefmt="fancy_grid"))

def  make_vector(matrix):
    return np.array(matrix).flatten()

#Parameters
r = 0.5
f = 10
dim = 7

def system(t,y): #system of ODEs
    global r,f,dim
    out = [0 for i in range(dim**2)]
    for j in range(dim):
        for k in range(dim):
            t = (j+1)*dim+k+1
            if t>=dim**2:
                out[j*dim+k] = -(r*(j+k)/2 + 1j*(j-k)*f)*y[j*dim+k]
            else:
                out[j*dim+k] = -(r*(j+k)/2 + 1j*(j-k)*f)*y[j*dim+k] + r*(( (j+1)*(k+1) )**0.5)*y[(j+1)*dim+k+1] 
    return out

k = 4 + 5j #Eigenvalue of the annihilation operator
D = np.array([[ (np.conj(k**m)*k**n)/(factorial(m)*factorial(n))**0.5 for m in range(dim) ] for n in range(dim)]) #Density matrix of a coherent state
y0 = make_vector(D)
print(D)
print("")
print("")

def make_matrix(vector):
    return np.array(vector).reshape(dim,dim)

def evolve_system(t,y0):
    sol = solve_ivp(system, [0,t] , y0 , dense_output=True)
    return make_matrix(sol.y[:,-1])


def compute_wigner(density_matrix):
    # Convert the density matrix to a Qobj, which is the type expected by qutip.wigner
    rho = Qobj(density_matrix)

    # Define the range of x and p values to compute the Wigner function over
    x = np.linspace(-5, 5, 200)
    p = np.linspace(-5, 5, 200)

    # Compute the Wigner function
    W = wigner(rho, x, p)

    return x, p, W


# for i in range(1,12,2):
#     x, p, W = compute_wigner(evolve_system(i,y0))
#     plt.contourf(x, p, W, 100)
#     plt.show()



fig, ax = plt.subplots()

# Initial Wigner function
x, p, W = compute_wigner(evolve_system(1, y0))

# Initial contour plot
contour = ax.contourf(x, p, W, 100)

# Update function for the animation
def update(i):
    ax.clear()
    x, p, W = compute_wigner(evolve_system(i, y0))
    contour = ax.contourf(x, p, W, 100)
    return contour,

T = np.arange(1, 15, 0.1)
# Create the animation
ani = animation.FuncAnimation(fig, update, frames=T, blit=True)

# Save the animation as a video file
ani.save('wigner_evolution_coherent_states.mp4', writer='ffmpeg')

plt.show()