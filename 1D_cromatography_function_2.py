import numpy as np
import matplotlib.pyplot as plt

# functions (first and second derivative on 1D grid)

def first_derivative(f, df, dx):
    # forward difference for the first point
    df[0] = (f[1] - f[0]) / dx

    for i in range(1, len(f) - 1):
        df[i] = (f[i + 1] - f[i - 1]) / (2 * dx)
    
    # backward difference for the last point
    df[-1] = (f[-1] - f[-2]) / dx

    return df

def second_derivative(f, d2f, dx):
    # forward difference for the first point
    d2f[0] = (f[2] - 2 * f[1] + f[0]) / dx**2

    for i in range(1, len(f) - 1):
        d2f[i] = (f[i + 1] - 2 * f[i] + f[i - 1]) / dx**2
    
    # backward difference for the last point
    d2f[-1] = (f[-1] - 2 * f[-2] + f[-3]) / dx**2

    return d2f

def time_derivative(f, df, d2f, f_p, param):
    vel = param[0]
    diffusion = param[1]
    absorption = param[2]

    term1 = -vel * df
    term2 = diffusion * d2f
    term3 = -absorption * (f-f_p)

    return term1 + term2 + term3

def time_derivative_p(f, fp, dq, constant):

    return constant * (f - fp) - dq

def time_derivative_q(q, q_star, const):
    return const * (q_star - q)

def get_q_star(conc, A, B, pH, n):
    return A * B *(pH/7)**n * conc / (1 + B*(pH/7)**n * conc)

def time_derivative_p_complicate(f, df, d2f, r, dq, param):
    diffusion = param[0]

    term1 = diffusion * d2f
    term2 = diffusion * 2 / r * df
    term3 = -dq

    return term1 + term2 + term3

def time_step(f, df, dt):
    return f + dt * df


def time_propagation_sol(conc_soli, zac):
    for i in range(3):
        conc_soli[i, 0, -1] = conc_soli[i, 0, -2]
        first_derivative(conc_soli[i, 0], conc_soli[i, 1], dx)
        second_derivative(conc_soli[i, 0], conc_soli[i, 2], dx)
        time_der = time_derivative(conc_soli[i,0], conc_soli[i,1], conc_soli[i,2], 0, (u, D_soli[0], 0))   
        conc_soli[i, 0] = time_step(conc_soli[i,0], time_der, dt)
    for i in range(3):
        conc_soli[i, 0, 0] = zac[i]
    return conc_soli

def racunanje_pH(conc_soli, K_eq):
    c_h = K_eq * conc_soli[1, 0] / conc_soli[2, 0]
    pH = -np.log10(c_h)
    return pH



def propagate(const, const_sol, u, D, alpha, absorb_rate, k_q, A_q, B_q, t_0, t_max, delez_smole,zac_conc,zac_soli, dt):
    N_species = len(alpha)
    dx = 1/(const.shape[3]-1)
    t = t_0
    while t < t_max:
        time_propagation_sol(conc_soli, zac_soli)

        for i in range(N_species):
            # Boundary conditions
            const[i, 0, 0, -1] = const[i, 0, 0, -2]

            # update derivatives
            first_derivative(const[i, 0, 0], const[i, 0, 1], dx)
            second_derivative(const[i, 0, 0], const[i, 0, 2], dx)
            # inter
            first_derivative(const[i, 1, 0], const[i, 1, 1], dx)
            second_derivative(const[i, 1, 0], const[i, 1, 2], dx)

        for i in range(N_species):
            time_der = time_derivative(conc[i, 0, 0], conc[i, 0, 1], conc[i, 0, 2], conc[i, 1, 0], (u, D[i], alpha[i]))
            time_der_inter = time_derivative_p(conc[i, 0, 0], conc[i, 1, 0], conc[i, 2, 1], absorb_rate[i])
            pH = racunanje_pH(const_sol, K_eq)
            q_star = get_q_star(conc[i, 0, 0], A_q, B_q, pH, n)
            time_der_q = time_derivative_q(conc[i, 2, 0], q_star, k_q)
            conc[i, 2, 1] = time_der_q
            outflow[i] += u * conc[i, 0, 0, -1] * dt
            all_outflows[i].append(outflow[i])

            # update concentrations
            conc[i, 0, 0] = time_step(conc[i, 0, 0], time_der, dt)
            conc[i, 1, 0] = time_step(conc[i, 1, 0], time_der_inter, dt)
            conc[i, 2, 0] = time_step(conc[i, 2, 0], time_der_q, dt)

            # concentration at zero
            conc[i, 0, 0, 0] = zac_conc[i] - outflow[i]/(1-delez_smole) - np.sum(conc[i, 0, 0, 1:]) * dx/(1-delez_smole) - np.sum(conc[i, 1, 0]) * dx/delez_smole - np.sum(conc[i, 2, 0]) * dx/delez_smole
            all_conc_0[i].append(conc[i, 0, 0, 0])
            all_conc_out[i].append(conc[i, 0, 0, -1])

        t += dt
    return all_outflows




# quantities: two concetrations and their first and second derivatives on 1D grid

n = 25
x = np.linspace(0, 1, n)
dx = x[1] - x[0]
N_species = 2
# vrsta, medij, odvod, pozicija
conc = np.zeros((N_species, 3, 3, n))
conc_soli = np.zeros((3,3,n))
outflow = np.zeros((N_species))
all_conc_0 = [[] for _ in range(N_species)]
all_outflows = [[] for _ in range(N_species)]
# concetration throughout the simulation at the end of the column
all_conc_out = [[] for _ in range(N_species)]


u = 0.2 #hitrost medija itzven kolone
D = (0.02, 0.01)
D_soli = (0.005, 0.005, 0.001) # last 2: no diffusion (Cl-, A-) ne rabim v preprostem modelu
n = 10 # pri računanju q_star
delez_smole = 0.5
absorb_rate = (0.02, 0.01)

alpha = []
for i in range(N_species):
    alpha.append(absorb_rate[i]*delez_smole/(1-delez_smole))

#dejanska hitrost medija: v koloni lahko večja zaradi smole ki zasede veloko prostora
u = u/(1-delez_smole)
zac_conc = (1, 1) # zacetna koncentracija proteinov

k_q = 0.01 # hitrost vezave na smolo
B_q = 1.0 # neki odvisno od ph
A_q = 1.0 #max koliko se veže na smolo
eps = 1.0 #

# Na+ Equilibrium Parameters
K_eq = 0.00001 # disociacijska konstanta za kislino

# Initial Conditions
conc[0, 0, 0, 0] = zac_conc[0]   # Protein A
conc[1, 0, 0, 0] = zac_conc[1]   # Protein B
    # Na+

# tu začnemo s stacionarnim stanjem
koncentracija_soli = 0.2
koncentracija_kisline = 0.1
conc_soli[0, 0] = koncentracija_soli    # Na+
conc_soli[1, 0] = koncentracija_soli    # Acetate (A-)
conc_soli[2, 0] = koncentracija_kisline # koncentracija kisline

# ---- Simulation Loop ----

dt = 0.0001
t = 0
t_max = 10.0


# parameters
u = 0.1
D = (0.01, 0.01)
absorb_rate = (0.2, 0.001)
q_star = 1
delez_smole = 0.8
alpha = tuple()
zac_conc = (1,1) 
for i in absorb_rate:
    rate = (delez_smole) / (1-delez_smole) * i
    alpha += (rate,) 
k_q = 0.01
B_q = 1.0
A_q = 1.0 # a je maks 1


# initial conditions
conc[:, 0, 0, 0] = 1

# times
dt = 0.00017
t_0 = 0
t_max = 20.0
zac_soli = [conc_soli[0,0,0], conc_soli[1,0,0], conc_soli[2,0,0]]
print(zac_soli)
# vse se uravnava tukaj, propagate vrne to kar gre ven in posodobi seznam koncentracij
outflows = propagate(conc,conc_soli, u, D, alpha, absorb_rate, k_q, A_q, B_q, t_0, t_max,delez_smole,zac_conc,  zac_soli, dt)

# plot
fig, ax = plt.subplots()
for i in range(N_species):
    ax.plot(outflows[i], label=f'Species {i+1}')

ax.set_xlabel('Time')
ax.set_ylabel('Outflow')
plt.show()

    