import os
import numpy as np
# from fitness import fitness, evaluate_baseline, DEFAULT_TAU, PENALTY_WEIGHT
from fitness import (
    fitness,
    evaluate_baseline,
    find_problematic_pairs,
    DEFAULT_TAU,
    PENALTY_WEIGHT,
)



POP_SIZE        = 100      
N_GENERATIONS   = 200     
ELITE_FRACTION  = 0.10    
MUTATION_RATE   = 0.25    
MUTATION_SIGMA  = 3.0                     
TOURNAMENT_SIZE = 4        
 
LAB_MIN = np.array([  0.0, -128.0, -128.0])
LAB_MAX = np.array([100.0,  127.0,  127.0])




def get_mutable_indices(original_lab,
                        threshold,
                        cvd_types=("protan", "deutan", "tritan")):

    remaining_pairs = set(
        find_problematic_pairs(
            original_lab,
            threshold=threshold,
            cvd_types=cvd_types,
        )
    )

    mutable = set()

    while remaining_pairs:

        # Count how many remaining conflicts each colour participates in
        counts = {}

        for i, j in remaining_pairs:
            counts[i] = counts.get(i, 0) + 1
            counts[j] = counts.get(j, 0) + 1

        # Pick the colour involved in the most conflicts
        chosen = max(counts, key=counts.get)

        mutable.add(chosen)

        # Remove every conflict solved by moving this colour
        remaining_pairs = {
            pair
            for pair in remaining_pairs
            if chosen not in pair
        }

    return sorted(mutable)


def init_population(original_lab,
                    mutable_indices,
                    pop_size=POP_SIZE,
                    sigma=MUTATION_SIGMA):

    population = []

    population.append(original_lab.copy())

    for _ in range(pop_size - 1):

        candidate = original_lab.copy()

        noise = np.random.normal(
            0,
            sigma,
            size=(len(mutable_indices), 3)
        )

        for idx, colour_idx in enumerate(mutable_indices):
            candidate[colour_idx] = np.clip(
                candidate[colour_idx] + noise[idx],
                LAB_MIN,
                LAB_MAX,
            )

        population.append(candidate)

    return population

def tournament_select(population, scores, tournament_size=TOURNAMENT_SIZE):

    indices = np.random.choice(len(population), size=tournament_size, replace=False)
    best_idx = indices[np.argmax([scores[i] for i in indices])]
    return population[best_idx].copy()
 
 
def crossover(parent_a,
              parent_b,
              mutable_indices):

    child = parent_a.copy()

    for idx in mutable_indices:

        if np.random.rand() < 0.5:
            child[idx] = parent_b[idx]

    return child


def mutate(individual, original_lab,
           mutable_indices, mutation_rate=MUTATION_RATE,
           sigma=MUTATION_SIGMA, tau=DEFAULT_TAU):

    mutated = individual.copy()
    k = len(mutated)
 
    for i in mutable_indices:
        if np.random.rand() < mutation_rate:
            noise = np.random.normal(0, sigma, size=3)
            new_colour = np.clip(mutated[i] + noise, LAB_MIN, LAB_MAX)
 
            # Check drift from original — if it exceeds tau, scale it back
            from cvd_module import delta_e
            drift = delta_e(original_lab[i], new_colour)
            if drift > tau:
                # Interpolate back toward original until drift = tau
                t = tau / drift          # 0 < t < 1
                new_colour = original_lab[i] + t * (new_colour - original_lab[i])
                new_colour = np.clip(new_colour, LAB_MIN, LAB_MAX)
 
            mutated[i] = new_colour
 
    return mutated
 
def run_ga(original_lab,
           mutable_indices,
           cvd_types=None,
           pop_size=POP_SIZE,
           n_generations=N_GENERATIONS,
           elite_fraction=ELITE_FRACTION,
           mutation_rate=MUTATION_RATE,
           mutation_sigma=MUTATION_SIGMA,
           tau=DEFAULT_TAU,
           verbose=True,
           snapshot_dir=None,
           snapshot_every=20):
    
    if cvd_types is None:
        cvd_types = ["protan", "deutan", "tritan"]
 
    k = len(original_lab)
    n_elite = max(1, int(pop_size * elite_fraction))
 
    # ── Print baseline so you can see how much improvement you get ──
    if verbose:
        print("\n" + "=" * 55)
        print("  GENETIC ALGORITHM — Colour Palette Optimisation")
        print("=" * 55)
        print(f"  Palette size : {k} colours")
        print(f"  Population   : {pop_size}")
        print(f"  Generations  : {n_generations}")
        print(f"  CVD types    : {cvd_types}")
        print(f"  Fidelity τ   : {tau} ΔE units")
        print("=" * 55 + "\n")
        evaluate_baseline(original_lab, cvd_types=cvd_types)
        
        print()
 
    # ── Step 1: Initialise population ──
    population = init_population(
        original_lab,
        mutable_indices,
        pop_size=pop_size,
        sigma=mutation_sigma,
    )
 
    # ── Snapshot setup ──
    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)
        from visualize import show_palette_comparison
 
    history = []        # best fitness per generation
    best_score = -np.inf
    best_palette = original_lab.copy()
 
    for gen in range(n_generations):
 
        # ── Step 2: Evaluate fitness of every individual ──
        scores = [
            fitness(ind, original_lab, cvd_types=cvd_types,
                    tau=tau, penalty_weight=PENALTY_WEIGHT)
            for ind in population
        ]
 
        # Track global best
        gen_best_idx = int(np.argmax(scores))
        gen_best_score = scores[gen_best_idx]
 
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_palette = population[gen_best_idx].copy()
 
        history.append(best_score)
 
        if verbose and (gen % 20 == 0 or gen == n_generations - 1):
            print(f"  Gen {gen:>4d}/{n_generations}  |  "
                  f"Best fitness = {best_score:7.3f}  |  "
                  f"Pop best = {gen_best_score:7.3f}")
 
        # ── Save snapshot of best palette every N generations ──
        if snapshot_dir is not None and (gen % snapshot_every == 0 or gen == n_generations - 1):
            snap_path = os.path.join(snapshot_dir, f"gen{gen}.png")
            show_palette_comparison(
                original_lab, best_palette,
                cvd_type=cvd_types,
                save_path=snap_path
            )
            if verbose:
                print(f"    → Snapshot saved: {snap_path}")
 
        # ── Step 3: Build next generation ──
        # Sort population by score (best first)
        sorted_pairs = sorted(zip(scores, range(pop_size)), reverse=True)
        sorted_indices = [idx for _, idx in sorted_pairs]
 
        next_gen = []
 
        # Elitism — copy top N directly, no changes
        for idx in sorted_indices[:n_elite]:
            next_gen.append(population[idx].copy())
 
        # Fill the rest with crossover + mutation
        while len(next_gen) < pop_size:
            parent_a = tournament_select(population, scores)
            parent_b = tournament_select(population, scores)
 
            child = crossover( parent_a, parent_b, mutable_indices,)
            child = mutate(
                child,
                original_lab,
                mutable_indices,
                mutation_rate=mutation_rate,
                sigma=mutation_sigma,
                tau=tau,
            )
            next_gen.append(child)
 
        population = next_gen
 
    if verbose:
        print(f"\n  ✓ Done. Best fitness = {best_score:.3f}\n")
 
    return best_palette, history
 
 

def report_result(original_lab, optimized_lab, cvd_types=None):

    if cvd_types is None:
        cvd_types = ["protan", "deutan", "tritan"]
 
    from fitness import fitness
    from cvd_module import delta_e
 
    print("\n" + "=" * 55)
    print("  RESULTS: Original vs Optimised Palette")
    print("=" * 55)
 
    orig_score, orig_details = fitness(
        original_lab, original_lab, cvd_types=cvd_types, return_details=True)
    opt_score, opt_details = fitness(
        optimized_lab, original_lab, cvd_types=cvd_types, return_details=True)
 
    print(f"\n  {'Metric':<30} {'Original':>10} {'Optimised':>10}")
    print("  " + "-" * 52)
    print(f"  {'Overall fitness':<30} {orig_score:>10.3f} {opt_score:>10.3f}")
    print(f"  {'Feasible (no constraint violation)':<30} "
          f"{'Yes' if orig_details['feasible'] else 'No':>10} "
          f"{'Yes' if opt_details['feasible'] else 'No':>10}")
 
    for cvd in cvd_types:
        label = f"Min ΔE ({cvd})"
        orig_de = orig_details["min_de_per_cvd"][cvd]
        opt_de  = opt_details["min_de_per_cvd"][cvd]
        print(f"  {label:<30} {orig_de:>10.3f} {opt_de:>10.3f}")
 
    print(f"\n  Per-colour drift from original:")
    for i, drift in enumerate(opt_details["drifts"]):
        print(f"    Colour {i}: ΔE = {drift:.2f}")
 
    print("=" * 55 + "\n")