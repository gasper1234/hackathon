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


def time_propagation_sol(conc_soli, zac,u):
    for i in range(2):
        conc_soli[i, 0, -1] = conc_soli[i, 0, -2]
        first_derivative(conc_soli[i, 0], conc_soli[i, 1], dx)
        second_derivative(conc_soli[i, 0], conc_soli[i, 2], dx)
        time_der = time_derivative(conc_soli[i,0], conc_soli[i,1], conc_soli[i,2], 0, (u, D_soli[0], 0))   
        conc_soli[i, 0] = time_step(conc_soli[i,0], time_der, dt)
    for i in range(2):
        conc_soli[i, 0, 0] = zac[i]
    return conc_soli

def racunanje_pH(conc_soli, K_eq):
    c_h = K_eq * conc_soli[0, 0] / conc_soli[1, 0]
    pH = -np.log10(c_h)
    return pH



def propagate(const, const_sol, u_array, D, alpha, absorb_rate, k_q, A_q, B_q, 
              t_0, t_max, delez_smole, zac_conc, zac_soli_array, dt):
    N_species = len(alpha)
    outflow = np.zeros((N_species))
    all_conc_0 = [[] for _ in range(N_species)]
    all_outflows = [[] for _ in range(N_species)]
    # concetration throughout the simulation at the end of the column
    all_conc_out = [[] for _ in range(N_species)]
    dx = 1/(const.shape[3]-1)
    t = t_0

    time_index = int(t)  # Get index for current hour
    zac_soli = zac_soli_array[time_index]
    u = u_array[time_index]

    while t < t_max:
        time_propagation_sol(conc_soli, zac_soli, u)

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
            conc[i, 0, 0, 0] = zac_conc[i] - outflow[i] - np.sum(conc[i, 0, 0, 1:]) * dx/(1-delez_smole) - np.sum(conc[i, 1, 0]) * dx/delez_smole - np.sum(conc[i, 2, 0]) * dx/delez_smole
            all_conc_0[i].append(conc[i, 0, 0, 0])
            all_conc_out[i].append(conc[i, 0, 0, -1])

        t += dt
        ucinkovitost = np.zeros_like(all_outflows)
        for i in range(N_species):
            ucinkovitost[i] = all_outflows[i] / (np.ones_like(all_outflows[i]) * zac_conc[i])

        zelen_protein = all_outflows[0]
        vsi_proteini = np.zeros_like(all_outflows[0])
        for i in range(N_species):
            vsi_proteini +=  all_outflows[i]
        cistost = zelen_protein/vsi_proteini


    return (all_outflows, ucinkovitost, cistost)




# quantities: two concetrations and their first and second derivatives on 1D grid

n = 25 #število krajevnih korakov
x = np.linspace(0, 1, n)
dx = x[1] - x[0]
N_species = 2 #št proteinov oopazovanih
# vrsta, medij, odvod, pozicija
conc = np.zeros((N_species, 3, 3, n)) #koncentracije proteinov v koloni
conc_soli = np.zeros((3,3,n)) #koncentracie ionov in kisline v koloni

D = (0.02, 0.01) # difuzijski koeficienti soli
D_soli = (0.005, 0.005) # last 2: no diffusion (Cl-, A-) ne rabim v preprostem modelu
n = 10 # pri računanju q_star
delez_smole = 0.5
absorb_rate = (0.02, 0.01) #za absorbcijo proteinov na smolo

alpha = []
for i in range(N_species):
    alpha.append(absorb_rate[i]*delez_smole/(1-delez_smole))

#dejanska hitrost medija: v koloni lahko večja zaradi smole ki zasede veloko prostora

zac_conc = (1, 3) # zacetna koncentracija proteinov

k_q = 0.01 # hitrost vezave na smolo
B_q = 1.0 # neki 
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

conc_soli[0, 0] = koncentracija_soli    # sol
conc_soli[1, 0] = koncentracija_kisline # koncentracija kisline

# times
dt = 0.0002
t_0 = 0
t_max = 2.0
#/(1-delez_smole)
u_array = np.linspace(0.2, 1.7, int(t_max))/(1-delez_smole)  # hitrosti sem dal da se lahko spremenijo enkrat na časovno enoto tukaj določiš profil

arr1 = np.tile(np.array([[0.2, 0.1]]), (int(t_max)//2, 1))
arr2 = np.tile(np.array([[0.3, 0.05]]), (int(t_max)//2, 1))

zac_soli_array = np.vstack((arr1, arr2))

#zac_soli_array = np.tile(np.array([[0.2, 0.1]]), (int(t_max), 1)) # enkrat na časovno enoto lahko spremeniš koncentracije ionov in kisline na začetku kolone

# vse se uravnava tukaj, propagate vrne to kar gre ven in posodobi seznam koncentracij
outflows = propagate(conc,conc_soli, u_array, D, alpha, absorb_rate, k_q, A_q, B_q, t_0, t_max,delez_smole,zac_conc,  zac_soli_array, dt)



# plot
fig, ax = plt.subplots(3,1)
for i in range(N_species):
    ax[0].plot(outflows[0][i], label=f'Species {i+1}')
    ax[1].plot(outflows[1][i], label=f'Species {i+1}')
ax[2].plot(outflows[2])

ax[0].set_xlabel('Time')
ax[0].set_ylabel('Outflow')
ax[1].set_ylabel('yield')
plt.show()
    
