import numpy as np
from qutip import Qobj, wigner,destroy
import matplotlib.pyplot as plt
from tabulate import tabulate 
import matplotlib.animation as animation
from math import factorial,sqrt

def lindblad_system(hamil,jump_ops,rho):
    out = -(hamil@rho-rho@hamil)*1j
    for op in jump_ops:
        out += op@rho@(op.H)-( (op.H)@op@rho + rho@(op.H)@op )/2
    return out


def evolve_system(hamil,jump_ops,rho_init,final_t,n_iter = 50):
    out = rho_init
    t = 0
    dt = final_t/n_iter
    while t<final_t:
        k1 = lindblad_system(hamil,jump_ops,out)*dt
        k2 = lindblad_system(hamil,jump_ops,out+0.5*k1)*dt
        k3 = lindblad_system(hamil,jump_ops,out+0.5*k2)*dt
        k4 = lindblad_system(hamil,jump_ops,out+k3)*dt
        out+=(k1 + 2*k2 + 2*k3 + k4)/6
        t+= dt
    return out

#An Example Case

nmax = 5
omega = 0

a_op = np.matrix(destroy(nmax).full())
hamil = omega*(a_op.H)@a_op
jump_ops = [2*a_op,1j*(np.eye(nmax)-a_op@a_op+(a_op.H)@(a_op.H))/2]

k = 3 + 0j #Initialising as a coherent state
D_coherent = np.array([[ (np.conj(k**m)*k**n)/(factorial(m)*factorial(n))**0.5 for m in range(nmax) ] for n in range(nmax)]) #Density matrix of a coherent state

n = 4 #Inititalising as a fock state
D_n = np.zeros((nmax,nmax),dtype=complex)
D_n[n][n]=1

def compute_wigner(density_matrix):
    rho = Qobj(density_matrix)

    x = np.linspace(-7, 7, 100)
    p = np.linspace(-7, 7, 100)

    W = wigner(rho, x, p)
    return x, p, W

i=0
while i<=1:
    print(tabulate(evolve_system(hamil,jump_ops,D_n,i,100),tablefmt="grid"))
    x, p, W = compute_wigner(evolve_system(hamil,jump_ops,D_n,i,100))
    plt.contourf(x, p, W, 100)
    plt.show()
    i+=0.1
