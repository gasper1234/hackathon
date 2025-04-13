import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Optimizirane funkcije za odvod, ki delujejo v vektorizirani obliki
# =============================================================================
def first_derivative_vec(f, dx):
    """
    Vektorizirana funkcija za izračun prvega odvoda.
    f: vhodno polje (zadnja dimenzija = prostor)
    dx: razdalja med točkami
    Vrne: novo polje prvega odvoda vzdolž zadnje dimenzije.
    """
    df = np.empty_like(f)
    df[..., 0] = (f[..., 1] - f[..., 0]) / dx
    df[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2 * dx)
    df[..., -1] = (f[..., -1] - f[..., -2]) / dx
    return df


def second_derivative_vec(f, dx):
    """
    Vektorizirana funkcija za izračun drugega odvoda.
    f: vhodno polje (zadnja dimenzija = prostor)
    dx: korak v prostoru
    Vrne: polje drugega odvoda vzdolž zadnje dimenzije.
    """
    d2f = np.empty_like(f)
    d2f[..., 0] = (f[..., 2] - 2 * f[..., 1] + f[..., 0]) / dx**2
    d2f[..., 1:-1] = (f[..., 2:] - 2 * f[..., 1:-1] + f[..., :-2]) / dx**2
    d2f[..., -1] = (f[..., -1] - 2 * f[..., -2] + f[..., -3]) / dx**2
    return d2f


# =============================================================================
# Ostale funkcije (skoraj nespremenjene)
# =============================================================================
def time_derivative(f, df, d2f, f_p, param):
    vel, diffusion, absorption = param
    term1 = -vel * df
    term2 = diffusion * d2f
    term3 = -absorption * (f - f_p)
    return term1 + term2 + term3


def time_derivative_p(f, fp, dq, constant):
    return constant * (f - fp) - dq


def time_derivative_q(q, q_star, const):
    return const * (q_star - q)


def get_q_star(conc, A, B, pH, n):
    return A * B * (pH / 7) ** n * conc / (1 + B * (pH / 7) ** n * conc)


def time_derivative_p_complicate(f, df, d2f, r, dq, param):
    diffusion = param[0]
    term1 = diffusion * d2f
    term2 = diffusion * 2 / r * df
    term3 = -dq
    return term1 + term2 + term3


def time_step(f, df, dt):
    return f + dt * df


def racunanje_pH(conc_soli, K_eq):
    c_h = K_eq * conc_soli[0, 0] / conc_soli[1, 0]
    pH = -np.log10(c_h)
    return pH


# =============================================================================
# Funkcija za evolucijo sistema
# =============================================================================
def evolve_system(
    conc,
    conc_soli,
    u_array,
    D,
    D_soli,
    alpha,
    absorb_rate,
    k_q,
    A_q,
    B_q,
    t_0,
    t_max,
    delez_smole,
    zac_conc,
    zac_soli_array,
    dt,
    n_ph,
    K_eq,
):
    """
    Izvede časovno evolucijo sistema z uporabo eksplicitnega Eulerjevega
    časovnega integratorja.

    Vhodni podatki:
      conc          - koncentracije proteinov, dimenzije (N_species, 3, 3, n)
                      (faza 0: tekočinski, faza 1: intermediat, faza 2: vezani na smolo)
      conc_soli     - koncentracije ionov/kisline, dimenzije (vrste, tip, prostor)
                      (npr. 0: sol, 1: kislina)
      u_array       - profil pretoka skozi kolon (ena vrednost na časovno enoto)
      D, D_soli     - difuzijski koeficienti za proteine in topilo
      alpha, absorb_rate, k_q, A_q, B_q, delez_smole, zac_conc, n_ph, K_eq – parametri modela
      zac_soli_array- vhodne koncentracije topila skozi čas (ena vrednost na časovno enoto)
      t_0, t_max, dt- časovni parametri

    Vrne slovar z evolucijskimi podatki:
      "conc"         : končna koncentracijska polja
      "conc_soli"    : končno stanje topila
      "all_outflows" : zgodovina odtokov za vsako vrsto (seznam listov)
      "all_conc_0"   : zgodovina koncentracij na vhodu (prva točka) za vsako vrsto
    """
    N_species = conc.shape[0]
    all_outflows = [[] for _ in range(N_species)]
    all_conc_0 = [[] for _ in range(N_species)]

    dx = 1 / (conc.shape[3] - 1)
    t = t_0

    while t < t_max:
        time_index = int(t)
        zac_soli = zac_soli_array[time_index]
        u = u_array[time_index]

        # Posodobi stanje topila (sol in kislina)
        conc_soli = time_propagation_sol(conc_soli, zac_soli, u, dx, dt, D_soli)

        # Vektorizirano posodabljanje mejnih pogojev in računanje odvoda za proteine
        conc[:, 0, 0, -1] = conc[:, 0, 0, -2]
        conc[:, 0, 1, :] = first_derivative_vec(conc[:, 0, 0, :], dx)
        conc[:, 0, 2, :] = second_derivative_vec(conc[:, 0, 0, :], dx)
        conc[:, 1, 1, :] = first_derivative_vec(conc[:, 1, 0, :], dx)
        conc[:, 1, 2, :] = second_derivative_vec(conc[:, 1, 0, :], dx)

        for i in range(N_species):
            # Časovni odvod za primarno fazo (tekočinski del)
            time_der = time_derivative(
                conc[i, 0, 0, :],
                conc[i, 0, 1, :],
                conc[i, 0, 2, :],
                conc[i, 1, 0, :],
                (u, D[i], alpha[i]),
            )
            # Časovni odvod za intermediat
            time_der_inter = time_derivative_p(
                conc[i, 0, 0, :], conc[i, 1, 0, :], conc[i, 2, 1, :], absorb_rate[i]
            )
            # Izračun pH iz stanja topila
            pH = racunanje_pH(conc_soli, K_eq)
            # Izračun q_star in časovni odvod za vezani na smolo del
            q_star = get_q_star(conc[i, 0, 0, :], A_q, B_q, pH, n_ph)
            time_der_q = time_derivative_q(conc[i, 2, 0, :], q_star, k_q)
            conc[i, 2, 1, :] = time_der_q

            # Posodobi odtok in zabeleži vrednosti
            delta_outflow = u * conc[i, 0, 0, -1] * dt
            # Posodobi akumulirani odtok za vsako vrsto
            if len(all_outflows[i]) == 0:
                all_outflows[i].append(delta_outflow)
            else:
                all_outflows[i].append(all_outflows[i][-1] + delta_outflow)

            # Časovni korak (ekspliti Eulerjev korak)
            conc[i, 0, 0, :] = time_step(conc[i, 0, 0, :], time_der, dt)
            conc[i, 1, 0, :] = time_step(conc[i, 1, 0, :], time_der_inter, dt)
            conc[i, 2, 0, :] = time_step(conc[i, 2, 0, :], time_der_q, dt)

            # Posodobi vhodni mejni pogoj: čeprav lahko fiksno določimo vhod,
            # če želite množični račun, se lahko uporabi formula.
            sum_term = (
                np.sum(conc[i, 0, 0, 1:]) * dx / (1 - delez_smole)
                + np.sum(conc[i, 1, 0, :]) * dx / delez_smole
                + np.sum(conc[i, 2, 0, :]) * dx / delez_smole
            )
            conc[i, 0, 0, 0] = zac_conc[i] - all_outflows[i][-1] - sum_term
            all_conc_0[i].append(conc[i, 0, 0, 0])

        t += dt

    return {
        "conc": conc,
        "conc_soli": conc_soli,
        "all_outflows": all_outflows,
        "all_conc_0": all_conc_0,
    }


# =============================================================================
# Funkcija za analizo odtokov
# =============================================================================
def compute_outflows(all_outflows, zac_conc, dt, t_max, required_purity, N_species):
    """
    Optimized compute_outflows funkcija, ki izvede isto analizo kot originalna,
    vendar uporablja vektorizirane operacije za hitrejši izračun.

    Parametri:
      all_outflows    - seznam, kjer je za vsako vrsto shranjena zgodovina
                        akumuliranega odtoka (seznam dolžine T, kjer T je število časovnih korakov)
      zac_conc        - začetne koncentracije (npr. tuple ali seznam)
      dt              - časovni korak
      t_max           - končni čas simulacije
      required_purity - zahtevana čistost (npr. 0.5)
      N_species       - število opazovanih vrst

    Vrne:
      koncno - matriko (oblike (num_intervals, 2)), kjer:
                - prva kolona vsebuje yield (končni kumulativni odtok, normaliziran z zac_conc)
                - druga kolona vsebuje seznam izbranih intervalnih številk (številke intervalov, ki skupaj dosegajo zahtevano čistost)
    """
    # Pretvorimo zgodovino odtokov v numpy array; predpostavljamo, da so vsi časovni podatki enake dolžine T.
    all_out_arr = np.array([np.array(o) for o in all_outflows])  # Oblika (N_species, T)
    T = all_out_arr.shape[1]

    # Določimo število intervalov – predpostavimo, da je dolžina vsakega intervala 1 časovna enota,
    # kar je enako int(1/dt), kar mora biti delitev časovnega sim. obdobja.
    interval_length = int(0.1 / dt)
    num_intervals = (
        T - 1
    ) // interval_length  # Lahko tudi: num_intervals = int(t_max) - 1

    # Vektorji, kjer bomo shranili za vsak interval:
    # - amount_green: odtok, ki pripada "zeleni" frakciji (za prvo vrsto)
    # - amount_total: skupni odtok (vsota po vseh vrstah)
    # - fraction: razmerje (čistost) v posameznem intervalu
    amount_green = np.empty(num_intervals)
    amount_total = np.empty(num_intervals)
    fractions = np.empty(num_intervals)
    indices = np.arange(num_intervals)

    for i in range(num_intervals):
        start = i * interval_length
        end = (i + 1) * interval_length
        # Od tok za prvo vrsto (zelena) in vsota odtokov (vse vrste)
        green_interval = all_out_arr[0, end] - all_out_arr[0, start]
        total_interval = np.sum(all_out_arr[:, end] - all_out_arr[:, start])
        amount_green[i] = green_interval
        amount_total[i] = total_interval
        fractions[i] = green_interval / (total_interval + 1e-12)

    # Za vsak interval (od začetka do i-tega intervala) zračunamo "kumulativno analizo"
    # tako, da upoštevamo samo intervale 0 .. i in s pomočjo vektorizacije
    koncno = np.zeros((num_intervals, 2), dtype=object)

    for i in range(num_intervals):
        # Uporabimo le intervale do i-tega (vključujoč)
        valid_frac = fractions[: i + 1]
        valid_amt = amount_green[: i + 1]
        valid_idx = indices[: i + 1]

        # Urejamo intervale glede na čistost v padajočem vrstnem redu
        sort_order = np.argsort(valid_frac)[::-1]
        sorted_frac = valid_frac[sort_order]
        sorted_amt = valid_amt[sort_order]
        sorted_idx = valid_idx[sort_order]

        # Vektorizirano kumulativno seštevanje:
        cum_amt = np.cumsum(sorted_amt)
        cum_weighted = np.cumsum(sorted_frac * sorted_amt)
        cum_purity = cum_weighted / (cum_amt + 1e-12)

        # Greedy: Izberemo intervale v danem urejenem zaporedju, dokler kumulativna čistost ostaja
        # nad zahtevano vrednostjo. Poiščemo prvi indeks, kjer se pogoj ne izpolni.
        below_threshold = np.where(cum_purity < required_purity)[0]
        if below_threshold.size > 0:
            j = below_threshold[0]
            if j == 0:
                selected_amt = 0.0
                selected_intervals = []
            else:
                selected_amt = cum_amt[j - 1]
                selected_intervals = sorted_idx[:j]
        else:
            selected_amt = cum_amt[-1]
            selected_intervals = sorted_idx

        # Normaliziramo yield z začetno koncentracijo in pretvorimo izbrane indekse v seštevek + 1
        koncno[i, 0] = selected_amt / zac_conc[0]
        # Ker želimo intervalne številke, jih povečamo za 1
        koncno[i, 1] = [int(x) + 1 for x in selected_intervals]

    return koncno


# =============================================================================
# Funkcija za posodobitev stanja topila (soli in kisline)
# =============================================================================
def time_propagation_sol(conc_soli, zac, u, dx, dt, D_soli):
    """
    Posodobi stanje topila (sol in kislina) z uporabo vektoriziranih odvodov.
    Uporabljamo le prve dve vrste (npr. Na+ in H+).
    """
    for i in range(2):
        conc_soli[i, 0, -1] = conc_soli[i, 0, -2]  # mejni pogoj na odtoku
        conc_soli[i, 1, :] = first_derivative_vec(conc_soli[i, 0, :], dx)
        conc_soli[i, 2, :] = second_derivative_vec(conc_soli[i, 0, :], dx)
        time_der = time_derivative(
            conc_soli[i, 0, :],
            conc_soli[i, 1, :],
            conc_soli[i, 2, :],
            0,  # f_p = 0
            (u, D_soli[0], 0),
        )
        conc_soli[i, 0, :] = time_step(conc_soli[i, 0, :], time_der, dt)
    for i in range(2):
        conc_soli[i, 0, 0] = zac[i]
    return conc_soli


# =============================================================================
# Parametri in inicializacija
# =============================================================================
n = 25  # Število krajevnih točk
x = np.linspace(0, 1, n)
dx = x[1] - x[0]

N_species = 3  # Število beljakovin (prvi je želeni, ostali so nečistoče)
# Struktura koncentracij: (beljakovine, faza, tip podatka, prostor)
# faze: 0 -> tekočinski; 1 -> intermediat; 2 -> vezani na smolo
# tipi podatkov: 0 -> količina, 1 -> 1. odvod, 2 -> 2. odvod
conc = np.zeros((N_species, 3, 3, n))
# Koncentracije ionov/kisline (dimenzije: (vrste, tip, prostor))
conc_soli = np.zeros((3, 3, n))

# Difuzijski koeficienti
D = (0.09, 0.01, 0.015)
D_soli = (0.1, 0.2)
n_ph = 10
delez_smole = 0.5
absorb_rate = (0.1, 0.02, 0.1)
alpha = [absorb_rate[i] * delez_smole / (1 - delez_smole) for i in range(N_species)]
zac_conc = (1, 1, 0.9)  # Začetna koncentracija beljakovin

k_q = 0.01
B_q = 1.0
A_q = 1.0
K_eq = 0.00001  # Disociacijska konstanta za kislino

# Inicializacija začetnih pogojev
conc[0, 0, 0, 0] = zac_conc[0]  # Protein A
conc[1, 0, 0, 0] = zac_conc[1]  # Protein B
koncentracija_soli = 0.2
koncentracija_kisline = 0.1
conc_soli[0, 0] = koncentracija_soli
conc_soli[1, 0] = koncentracija_kisline

# Časovni parametri
dt = 0.002
t_0 = 0
t_max = 30.0
required_purity = 0.7
u_array = np.linspace(0.1, 0.2, int(t_max)) / (1 - delez_smole)
arr1 = np.tile(np.array([[0.2, 0.1]]), (int(t_max) // 2, 1))
arr2 = np.tile(np.array([[0.3, 0.05]]), (int(t_max) // 2, 1))
zac_soli_array = np.vstack((arr1, arr2))
# Lahko uporabite tudi:
# zac_soli_array = np.tile(np.array([[0.2, 0.1]]), (int(t_max), 1))

# =============================================================================
# Izvedba evolucije sistema in analiza odtokov
# =============================================================================
# Najprej evolucija sistema
evolution_data = evolve_system(
    conc,
    conc_soli,
    u_array,
    D,
    D_soli,
    alpha,
    absorb_rate,
    k_q,
    A_q,
    B_q,
    t_0,
    t_max,
    delez_smole,
    zac_conc,
    zac_soli_array,
    dt,
    n_ph,
    K_eq,
)

if __name__ == "__main__":

    print(evolution_data["conc"].shape)

    # # Nato na podlagi zbrane zgodovine odtokov izvedemo analizo
    # koncno = compute_outflows(
    #     evolution_data["all_outflows"], zac_conc, dt, t_max, required_purity, N_species
    # )
    # print(koncno)

    # # Za prikaz grafa bomo pretvorili yield vrednosti v CPU in kot float.
    # yield_values = [koncno[i][0] for i in range(len(koncno))]
    # # Prikaz yield (yield values) glede na neki "interval index"
    # plt.plot(yield_values)
    # plt.xlabel("Interval index")
    # plt.ylabel("Yield (normaliziran)")
    # plt.title("Yield vrednosti")
    # plt.show()

    # purity = [0.05 * i for i in range(1, 21)]
    # # Izračunamo čistost za vsak interval
    # yield_values = []
    # for p in purity:
    #     koncno = compute_outflows(
    #         evolution_data["all_outflows"], zac_conc, dt, t_max, p, N_species
    #     )
    #     yield_values.append(koncno[-1, 0])

    # plt.plot(purity, yield_values)
    # plt.show()
