import numpy as np
import matplotlib.pyplot as plt

# ---- Derivative Functions ----

def first_derivative(f, df, dx):
    df[0] = (f[1] - f[0]) / dx
    for i in range(1, len(f) - 1):
        df[i] = (f[i + 1] - f[i - 1]) / (2 * dx)
    df[-1] = (f[-1] - f[-2]) / dx
    return df

def second_derivative(f, d2f, dx):
    d2f[0] = (f[2] - 2 * f[1] + f[0]) / dx**2
    for i in range(1, len(f) - 1):
        d2f[i] = (f[i + 1] - 2 * f[i] + f[i - 1]) / dx**2
    d2f[-1] = (f[-1] - 2 * f[-2] + f[-3]) / dx**2
    return d2f

def time_derivative(f, df, d2f, f_p, param):
    vel, diffusion, absorption = param
    return -vel * df + diffusion * d2f - absorption * (f - f_p)

def time_derivative_p(f, fp, dq, constant):
    return constant * (f - fp) - dq

def time_derivative_q(q, q_star, const):
    return const * (q_star - q)

def get_q_star(conc, A, B):
    return A * B * conc / (1 + B * conc)

def time_step(f, df, dt):
    return f + dt * df

# ---- Na+ Specific Functions ----

def calc_c_Na_eq(q_Na, c_H, K, q_R):
    return (q_Na**2 * c_H**2) / (K * (q_R - q_Na + 1e-8))  # avoid div-by-zero

def time_derivative_q_Na(q_Na, c_Na, c_Na_eq, k_in):
    return k_in * (c_Na - c_Na_eq)

# ---- Setup ----

n = 25
x = np.linspace(0, 1, n)
dx = x[1] - x[0]

N_species = 2  # A, B, Na+, Cl-, Acetate
conc = np.zeros((N_species, 3, 3, n))  # [species, type, derivative, position]
conc_soli = np.zeros((3,3,3,n))
outflow = np.zeros((N_species))
all_conc_0 = [[] for _ in range(N_species)]
all_conc_soli = [[],[],[]]
all_outflows = [[] for _ in range(N_species)]
all_conc_out = [[] for _ in range(N_species)]

# Parameters

u = 0.4 #hitrost medija itzven kolone
D = (0.02, 0.01)
#D_soli = (0.005, 0.005, 0.001)  # last 2: no diffusion (Cl-, A-) ne rabim v preprostem modelu

delez_smole = 0.7
absorb_rate = (0.2, 0.1)
absorb_rate_soli = (0.0, 0.0, 0.0)

alpha = []
alpha_soli = []
for i in range(N_species):
    alpha.append(absorb_rate[i]*delez_smole/(1-delez_smole))
for i in range(3):
    alpha_soli.append(absorb_rate_soli[i]*delez_smole/(1-delez_smole))

#dejanska hitrost medija: v koloni lahko večja zaradi smole ki zasede veloko prostora
u = u/(1-delez_smole)
zac = (1,1)

k_q = 0.01 # hitrost vezave na smolo
B_q = 1.0 # neki odvisno od ph
A_q = 1.0 #max koliko se veže na smolo
eps = 1.0 #

# Na+ Equilibrium Parameters
K_eq = 1.0 # disociacijska konstanta za na+
q_R = 1.0 # ionska kapaciteta liganda
k_in = 1e8 # hitrost vezave na+ na sol
k_a = 0.1# disociacijska konstanta za izračun cH+

# Initial Conditions
conc[0, 0, 0, 0] = zac[0]   # Protein A
conc[1, 0, 0, 0] = zac[1]   # Protein B
conc_soli[0, 0, 0] = 0.5    # Na+
conc_soli[1, 0, 0] = 0.5    # Cl-
conc_soli[2, 0, 0] = 0.5    # Acetate (A-)
c_H = k_a*(conc_soli[2,0,0]/(conc_soli[0, 0, 0]- conc_soli[1, 0, 0])-1)  #staccionarne rešitve
q_Na = conc_soli[0, 0, 0] # stacionarne rešitve
c_Na = conc_soli[0, 0, 0] # stacionarne rešitve
c_Na_eq = calc_c_Na_eq(q_Na, c_H, K_eq, q_R) # stacionarne rešitve
conc_soli[0, 2, 0] = c_Na_eq


# ---- Simulation Loop ----

dt = 0.0001
t = 0
t_max = 4.0

while t < t_max:
    for i in range(N_species):
        conc[i, 0, 0, -1] = conc[i, 0, 0, -2]

        first_derivative(conc[i, 0, 0], conc[i, 0, 1], dx)
        second_derivative(conc[i, 0, 0], conc[i, 0, 2], dx)
        first_derivative(conc[i, 1, 0], conc[i, 1, 1], dx)
        second_derivative(conc[i, 1, 0], conc[i, 1, 2], dx)

    # ---- Proteins A and B ----
    
        time_der = time_derivative(conc[i, 0, 0], conc[i, 0, 1], conc[i, 0, 2], conc[i, 1, 0], (u, D[i], alpha[i]))
        time_der_inter = time_derivative_p(conc[i, 0, 0], conc[i, 1, 0], conc[i, 2, 1], absorb_rate[i])
        q_star = get_q_star(conc[i, 0, 0], A_q, B_q)
        time_der_q = time_derivative_q(conc[i, 2, 0], q_star, k_q)
        conc[i, 2, 1] = time_der_q

        outflow[i] += u * conc[i, 0, 0, -1] * dt
        all_outflows[i].append(outflow[i])

        conc[i, 0, 0] = time_step(conc[i, 0, 0], time_der, dt)
        conc[i, 1, 0] = time_step(conc[i, 1, 0], time_der_inter, dt)
        conc[i, 2, 0] = time_step(conc[i, 2, 0], time_der_q, dt)

        conc[i, 0, 0, 0] = zac[i] - outflow[i]/(1-delez_smole) - np.sum(conc[i, 0, 0, 1:]) * dx/(1-delez_smole) - np.sum(conc[i, 1, 0]) * dx/delez_smole - np.sum(conc[i, 2, 0]) * dx/delez_smole
        all_conc_0[i].append(conc[i, 0, 0, 0])
        all_conc_out[i].append(conc[i, 0, 0, -1])

    # ---- Na+ (Salt) ----
    i = 0

    time_der_Na = -u * conc_soli[0, 0, 1] 
    c_H = k_a*(conc_soli[2,0,0]/(conc_soli[0, 0, 0]- conc_soli[1, 0, 0]+0.00000000000001)-1)  # Assume [H+] from protein A
    q_Na = conc_soli[0, 0, 0]
    c_Na = conc_soli[0, 0, 0]
    c_Na_eq = calc_c_Na_eq(q_Na, c_H, K_eq, q_R)
    dq_Na = time_derivative_q_Na(q_Na, c_Na, c_Na_eq, k_in)
    conc_soli[0, 2, 1] = dq_Na

    conc_soli[0, 0, 0] = time_step(c_Na, time_der_Na, dt)
    conc_soli[0, 2, 0] = time_step(q_Na, dq_Na, dt)



    # ---- Cl- and Acetate (A-) ----
    for i in [1, 2]:
        time_der = -u * conc_soli[i, 0, 1] 
        conc_soli[i, 0, 0] = time_step(conc_soli[i, 0, 0], time_der, dt)


    # Plot at end
    if t >= t_max - dt:
        fig, ax = plt.subplots(N_species, 1, figsize=(10, 2*N_species))
        for i in range(N_species):
            ax[i].plot(x, conc[i, 0, 0], label='liquid')
            if i < 3:
                ax[i].plot(x, conc[i, 1, 0], label='intermediate')
                ax[i].plot(x, conc[i, 2, 0], label='bound')
            ax[i].set_title(f"Species {i}")
            ax[i].legend()
        plt.tight_layout()
        plt.show()

    t += dt

# Final plots for inlet/outlet conc.
fig, ax = plt.subplots(2, 1)
for i in range(N_species):
    ax[0].plot(all_conc_out[i], label=f'species {i} out')
    ax[1].plot(all_conc_0[i], label=f'species {i} in')
ax[0].set_title('Outlet Concentrations')
ax[1].set_title('Inlet Concentrations')
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()

# ---- pH Calculation ----
z = [0, 0, +1, -1, -1]  # charges for A, B, Na+, Cl-, Acetate
A = 0.5  # Debye-Hückel constant (unit dependent)
b = 0.1  # Constant for log gamma formula
z_H = 1  # charge of H+

c_H_plus =  c_H = k_a*(conc_soli[2,0,0]/(conc_soli[0, 0, 0]- conc_soli[1, 0, 0]+0.00000000000001)-1)  # Assume [H+] from protein A
c_ions = [conc[n, 0, 0]  for n in range(N_species)] + [ conc_soli[i, 0, 0] for i in range(3)]

# Calculate Ionic Strength I
I = np.zeros_like(c_H_plus)
for i in range(N_species+3):
    I += 0.5 * z[i]**2 * c_ions[i]

# log10 gamma_H+
log10_gamma_H = - (2 * z_H**2 * A * np.sqrt(I)) / (1 + np.sqrt(I) - b * np.sqrt(I) + 1e-8)

# gamma_H+
gamma_H = 10 ** log10_gamma_H

# pH = -log10(gamma_H * c_H+)
pH = -np.log10(gamma_H * c_H_plus + 1e-12)  # prevent log(0)

# Plot pH
plt.figure(figsize=(8, 4))
plt.plot(x, pH, label='pH Profile', color='purple')
plt.title('pH Profile Along the Column')
plt.xlabel('Position x')
plt.ylabel('pH')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
