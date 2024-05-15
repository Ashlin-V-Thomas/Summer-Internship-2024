import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tabulate import tabulate   

# This program simulates the time evolution of density matrix of photons in a cavity with dissipation.


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


D = generate_random_density_matrix(dim)
y0 = make_vector(D)
print_matrix(D)
print("")
print("")

def make_matrix(vector):
    return np.array(vector).reshape(dim,dim)

def evolve_system(t,y0):
    sol = solve_ivp(system, [0,t] , y0 , dense_output=True)
    return make_matrix(sol.y[:,-1])


def plot_probability(t,y0):
    T = np.linspace(0, t, 100)
    sol = solve_ivp(system, [0,t] , y0 , dense_output=True)
    y = sol.sol(T)
    prob = np.array([np.abs(y[i])**2 for i in range(len(y))])
    plt.plot(T, prob.T)
    plt.show()

for i in range(1,15,2):
    print_matrix(evolve_system(i,y0))
    print("")
    print("")

