#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cma  # CMA-ES knjižnica
from concurrent.futures import ProcessPoolExecutor
import argparse

# -----------------------------------------------------------------------------
# Predpostavljamo, da imate definirane funkcije evolve_system in compute_outflows
# v datoteki optimized_1d.py; če jih nimate, jih morate vključiti.
# -----------------------------------------------------------------------------
from optimized_1d import evolve_system, compute_outflows


# -----------------------------------------------------------------------------
# Parameterizacija profilov z linearnim interpolacijskim modelom (samo numpy)
# -----------------------------------------------------------------------------
def generate_profiles_control(params, t_max, delez_smole, n_control_u, n_control_sol):
    """
    Iz parametričnega vektorja (dolžine n_control_u+2*n_control_sol) generira profile:
      - u_array: linearno interpoliran profil z n_control_u kontrolnimi točkami,
      - zac_soli_array: linearno interpoliran profil za vsako komponento topila s
                        n_control_sol kontrolnimi točkami (za vsako komponento).

    Vrne:
      u_array: numpy array oblike (T,)
      zac_soli_array: numpy array oblike (T, 2)
    """
    T = int(t_max)
    t_full = np.linspace(0, t_max, T)
    u_control = np.array(params[0:n_control_u])
    sol1_control = np.array(params[n_control_u : n_control_u + n_control_sol])
    sol2_control = np.array(
        params[n_control_u + n_control_sol : n_control_u + 2 * n_control_sol]
    )
    positions_u = np.linspace(0, t_max, n_control_u)
    positions_sol = np.linspace(0, t_max, n_control_sol)
    u_array = np.interp(t_full, positions_u, u_control) / (1 - delez_smole)
    sol1_array = np.interp(t_full, positions_sol, sol1_control)
    sol2_array = np.interp(t_full, positions_sol, sol2_control)
    zac_soli_array = np.column_stack((sol1_array, sol2_array))
    return u_array, zac_soli_array


# =============================================================================
# Objektivna funkcija za CMA-ES optimizacijo (samo numpy)
# =============================================================================
def objective(params, n_control_u, n_control_sol):
    """
    Objektivna funkcija simulacije.
    Parametrični vektor je dolg (n_control_u + 2*n_control_sol + 2):
      - Prvih n_control_u + 2*n_control_sol elementov so kontrolne točke za profile:
           [u_control, sol1_control, sol2_control].
      - Element: kandidat izbora, ki se skalira, zaokroži in clip-a (da se izbere indeks med 0 in 9).
      - Zadnji element: optimiran delež smole, omejen med 0.3 in 0.7.

    Za izbrani kandidat in optimiran delež smole poganjamo simulacijo in vrnemo negativni yield.
    """
    n_profile = n_control_u + 2 * n_control_sol

    # Kandidat izbora (element n_profile)
    candidate_param = params[n_profile] * 100  # Skaliranje
    candidate_index = int(np.clip(np.round(candidate_param), 0, 9))

    # Optimiran delež smole (zadnji element)
    optimized_delez_smole = params[n_profile + 1]

    # Kandidatski parametri – vzeti iz candidate_set
    candidate = candidate_set[candidate_index]  # [k_q, A_q, B_q, K_eq]
    k_q_candidate, A_q_candidate, B_q_candidate, K_eq_candidate = candidate

    # Profil parametri (prvih n_profile elementov)
    profile_params = params[0:n_profile]
    u_control = profile_params[0:n_control_u]
    sol1_control = profile_params[n_control_u : n_control_u + n_control_sol]
    sol2_control = profile_params[n_control_u + n_control_sol : n_profile]
    if not (
        np.all(u_control >= 0.01)
        and np.all(u_control <= 1.0)
        and np.all(sol1_control >= 0.0)
        and np.all(sol1_control <= 1.0)
        and np.all(sol2_control >= 0.0)
        and np.all(sol2_control <= 1.0)
    ):
        return 1e6

    # Generiramo profile
    u_array, zac_soli_array = generate_profiles_control(
        profile_params, t_max, optimized_delez_smole, n_control_u, n_control_sol
    )

    conc_sim = conc.copy()
    conc_soli_sim = conc_soli.copy()

    sim_data = evolve_system(
        conc_sim,
        conc_soli_sim,
        u_array,
        D,
        D_soli,
        alpha,
        absorb_rate,
        k_q_candidate,
        A_q_candidate,
        B_q_candidate,
        t_0,
        t_max,
        optimized_delez_smole,
        zac_conc,
        zac_soli_array,
        dt,
        n_ph,
        K_eq_candidate,
    )
    koncno = compute_outflows(
        sim_data["all_outflows"],
        zac_conc,
        dt,
        t_max,
        required_purity,
        N_species,
    )
    final_yield = koncno[-1][0]
    obj_value = -final_yield
    print(
        "Params:",
        params,
        "Candidate index:",
        candidate_index,
        "Delež smole:",
        optimized_delez_smole,
        "Yield:",
        final_yield,
    )
    return obj_value


# =============================================================================
# Paralelizirana evaluacija objektivne funkcije (samo numpy)
# =============================================================================

from functools import partial


def evaluate_solution(sol, n_control_u, n_control_sol):
    return objective(sol, n_control_u, n_control_sol)


def parallel_objective(solutions, num_workers, n_control_u, n_control_sol):
    func = partial(
        evaluate_solution, n_control_u=n_control_u, n_control_sol=n_control_sol
    )
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(func, solutions))
    return results


# =============================================================================
# Glavni program – konfiguracija preko argparse (primer za SLURM)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Optimizacija profilov in parametrov s CMA-ES."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Število procesov za vzporedno evalvacijo",
    )
    parser.add_argument(
        "--max_iter", type=int, default=20, help="Maksimalno število iteracij CMA-ES"
    )
    args = parser.parse_args()
    num_workers = args.num_workers
    max_iter = args.max_iter

    # -----------------------------------------------------------------------------
    # Inicializacija kandidat seta (10 kandidatov, vsak [k_q, A_q, B_q, K_eq])
    # -----------------------------------------------------------------------------
    global candidate_set
    candidate_basis = np.array([0.01, 1.0, 1.0, 1e-5])
    np.random.seed(42)
    candidate_set = candidate_basis * (1 + np.random.uniform(-0.2, 0.2, size=(10, 4)))
    print("Kandidat set:\n", candidate_set)

    # -----------------------------------------------------------------------------
    # Inicializacija osnovnih parametrov (samo numpy)
    # -----------------------------------------------------------------------------
    global n, x, dx, N_species, conc, conc_soli, D, D_soli, n_ph, delez_smole_global, absorb_rate, alpha, zac_conc, k_q, B_q, A_q, K_eq, dt, t_0, t_max, required_purity
    n = 25
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    N_species = 3
    conc = np.zeros((N_species, 3, 3, n), dtype=np.float32)
    conc_soli = np.zeros((3, 3, n), dtype=np.float32)

    D = (0.09, 0.01, 0.015)
    D_soli = (0.1, 0.2)
    n_ph = 10
    delez_smole_global = 0.5  # Osnovna vrednost (se bo optimizirala)
    absorb_rate = (0.1, 0.02, 0.1)
    alpha = [
        absorb_rate[i] * delez_smole_global / (1 - delez_smole_global)
        for i in range(N_species)
    ]
    zac_conc = (1, 1, 0.9)
    k_q = 100
    B_q = 0.00001
    A_q = 1.0
    K_eq = 0.00001
    dt = 0.002
    t_0 = 0.0
    t_max = 30.0
    required_purity = 0.6

    conc[0, 0, 0, 0] = zac_conc[0]
    conc[1, 0, 0, 0] = zac_conc[1]
    koncentracija_soli = 0.2
    koncentracija_kisline = 0.1
    conc_soli[0, 0] = koncentracija_soli
    conc_soli[1, 0] = koncentracija_kisline

    # -----------------------------------------------------------------------------
    # Določitev števila kontrolnih točk (lahko jih prilagodite, s spreminjanjem začetnih vektorjev)
    # -----------------------------------------------------------------------------
    # Spremenite ta začetna vektorja, če želite drugačno število kontrolnih točk.
    u_initial = np.array([0.1, 0.2, 0.3, 0.4])  # kontrolne točke za u_array
    sol1_initial = np.array([0.1, 0.2, 0.3, 0.4])  # kontrolne točke za sol1
    sol2_initial = np.array([0.1, 0.2, 0.3, 0.4])  # kontrolne točke za sol2

    n_control_u = len(u_initial)
    n_control_sol = len(sol1_initial)

    # -----------------------------------------------------------------------------
    # Inicializacija optimizacijskega vektorja
    # -----------------------------------------------------------------------------
    profile_init = np.concatenate((u_initial, sol1_initial, sol2_initial))
    candidate_choice_init = np.array([0.04])  # začetni kandidat (0.04*100 = 4)
    optimized_smole_init = np.array([0.5])  # začetni delež smole
    init_params = np.concatenate(
        (profile_init, candidate_choice_init, optimized_smole_init)
    )
    # Dolžina optimizacijskega vektorja = n_control_u + 2*n_control_sol + 2

    # -----------------------------------------------------------------------------
    # Definicija bounds za CMA-ES
    # -----------------------------------------------------------------------------
    profile_lower = np.concatenate(
        (
            np.full(n_control_u, 0.01),
            np.full(n_control_sol, 0.0),
            np.full(n_control_sol, 0.0),
        )
    )
    profile_upper = np.concatenate(
        (
            np.full(n_control_u, 1.0),
            np.full(n_control_sol, 1.0),
            np.full(n_control_sol, 1.0),
        )
    )
    candidate_lower = np.array([0.0])
    candidate_upper = np.array([0.09])
    smole_lower = np.array([0.3])
    smole_upper = np.array([0.7])
    lower_bounds = np.concatenate((profile_lower, candidate_lower, smole_lower))
    upper_bounds = np.concatenate((profile_upper, candidate_upper, smole_upper))

    es = cma.CMAEvolutionStrategy(
        init_params,
        0.05,
        {"bounds": [lower_bounds.tolist(), upper_bounds.tolist()], "maxiter": max_iter},
    )

    # -----------------------------------------------------------------------------
    # CMA-ES optimizacijski cikel s paralelizacijo
    # -----------------------------------------------------------------------------
    while not es.stop():
        solutions = es.ask()
        obj_values = parallel_objective(
            solutions, num_workers, n_control_u, n_control_sol
        )
        es.tell(solutions, obj_values)
        es.disp()

    best_params = es.result.xbest
    print("Najboljši parametri:", best_params)

    # -----------------------------------------------------------------------------
    # Izpis začetnih (neoptimiziranih) in optimiziranih vrednosti modela
    # -----------------------------------------------------------------------------
    print("\n------ Pregled modelskih parametrov ------")
    print("Osnovni modelski parametri:")
    print("  D:", D)
    print("  D_soli:", D_soli)
    print("  n_ph:", n_ph)
    print("  absorb_rate:", absorb_rate)
    print("  alpha:", alpha)
    print("  zac_conc:", zac_conc)
    print("  k_q:", k_q, "  A_q:", A_q, "  B_q:", B_q, "  K_eq:", K_eq)
    print("  dt:", dt, "t_0:", t_0, "t_max:", t_max)
    print("  required_purity:", required_purity)
    print("\nNeoptimizirani profili:")
    print("  u_control:", u_initial)
    print("  sol1_control:", sol1_initial)
    print("  sol2_control:", sol2_initial)
    print("  Kandidatski izbor (neoptimiziran):", candidate_choice_init[0])
    print("  Delež smole (neoptimiziran):", optimized_smole_init[0])
    print("\nOptimizirani parametri:")
    opt_profile = best_params[: n_control_u + 2 * n_control_sol]
    opt_u_array, opt_zac_soli_array = generate_profiles_control(
        opt_profile, t_max, best_params[-1], n_control_u, n_control_sol
    )
    print("  u_control (profil):", best_params[:n_control_u])
    print(
        "  sol1_control (profil):",
        best_params[n_control_u : n_control_u + n_control_sol],
    )
    print(
        "  sol2_control (profil):",
        best_params[n_control_u + n_control_sol : n_control_u + 2 * n_control_sol],
    )
    candidate_index = int(
        np.clip(np.round(best_params[n_control_u + 2 * n_control_sol] * 100), 0, 9)
    )
    opt_candidate = candidate_set[candidate_index]
    print("  Kandidatski izbor (index):", candidate_index)
    print("  Kandidatski parametri (k_q, A_q, B_q, K_eq):", opt_candidate)
    print("  Delež smole (optimiziran):", best_params[-1])

    # -----------------------------------------------------------------------------
    # Vizualizacija yield vrednosti (simulacija optimiziranih parametrov)
    # -----------------------------------------------------------------------------
    sim_data_best = evolve_system(
        conc.copy(),
        conc_soli.copy(),
        opt_u_array,
        D,
        D_soli,
        alpha,
        absorb_rate,
        opt_candidate[0],
        opt_candidate[1],
        opt_candidate[2],
        t_0,
        t_max,
        best_params[-1],
        zac_conc,
        opt_zac_soli_array,
        dt,
        n_ph,
        opt_candidate[3],
    )
    koncno_best = compute_outflows(
        sim_data_best["all_outflows"],
        zac_conc,
        dt,
        t_max,
        required_purity,
        N_species,
    )
    yield_values_best = [koncno_best[i][0] for i in range(len(koncno_best))]
    plt.plot(yield_values_best, marker="o")
    plt.xlabel("Interval index")
    plt.ylabel("Yield (normaliziran)")
    plt.title("Yield vrednosti, optimizirane s CMA-ES")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimizacija profilov in parametrov s CMA-ES."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Število procesov za vzporedno evalvacijo",
    )
    parser.add_argument(
        "--max_iter", type=int, default=20, help="Maksimalno število iteracij CMA-ES"
    )
    args = parser.parse_args()
    main()
