# import matplotlib.pyplot as plt
# import numpy as np

# from palette_extraction import extract_palette_from_path
# from ga import run_ga, report_result
# from cvd_module import lab_to_srgb


# # -----------------------------------
# # SETTINGS
# # -----------------------------------
# IMAGE_PATH = "test_chart.png"      # your input image
# K_COLORS   = 6                     # number of palette colors


# # -----------------------------------
# # STEP 1: Extract palette
# # -----------------------------------
# print("\nExtracting palette from image...\n")

# original_lab, original_rgb, labels, counts = extract_palette_from_path(
#     IMAGE_PATH,
#     k=K_COLORS
# )

# print("Original RGB Palette:")
# print(np.round(original_rgb, 3))


# # -----------------------------------
# # STEP 2: Run Genetic Algorithm
# # -----------------------------------
# best_lab, history = run_ga(original_lab)


# # -----------------------------------
# # STEP 3: Convert best palette to RGB
# # -----------------------------------
# best_rgb = lab_to_srgb(best_lab)

# print("\nOptimized RGB Palette:")
# print(np.round(best_rgb, 3))


# # -----------------------------------
# # STEP 4: Print comparison report
# # -----------------------------------
# report_result(original_lab, best_lab)


# # -----------------------------------
# # STEP 5: Plot fitness graph
# # -----------------------------------
# plt.figure(figsize=(8,5))
# plt.plot(history, linewidth=2)
# plt.title("Genetic Algorithm Progress")
# plt.xlabel("Generation")
# plt.ylabel("Best Fitness")
# plt.grid(True)
# plt.show()

"""
main.py
=======
Entry point. Runs the full pipeline end to end:
 
    1. Load an image  →  extract dominant colour palette
    2. Run the GA     →  optimise palette for CVD accessibility
    3. Show results   →  before/after swatches + convergence plot
 
USAGE
-----
With an image:
    python main.py --image path/to/your/image.png
 
Without an image (uses a built-in test palette):
    python main.py
 
OPTIONS
-------
    --image     Path to input image (PNG, JPG, etc.)
    --colours   Number of palette colours to extract (default: 6)
    --cvd       CVD types to optimise for (default: all three — protan deutan tritan)
    --tau       Fidelity threshold in ΔE units (default: 15)
    --gens      Number of GA generations (default: 200)
    --pop       GA population size (default: 60)
    --runs      Number of independent GA runs (default: 1)
    --save      Save output figures instead of just displaying them
"""
 
import argparse
import numpy as np
 
from cvd_module import srgb_to_lab, lab_to_srgb
from palette_extraction import extract_palette_from_path, print_palette
from ga import run_ga, report_result, POP_SIZE, N_GENERATIONS
from fitness import fitness, DEFAULT_TAU
from visualize import show_palette_comparison, show_convergence
 
 
# ─────────────────────────────────────────────────────────────
# Built-in test palette (used when no image is provided)
# A deliberately bad palette: all red-green confusion colours
# ─────────────────────────────────────────────────────────────
# TEST_PALETTE_RGB = np.array([
#     [0.55, 0.407, 0.121],   # red
#     [0.9170, 0.96, 0.67],   # green
#     [0.72940, 0.8117, 0.9333],   # orange
#     [0.8588, 0.3921, 0.8392],   # lime green
#     [0.2156, 0.91764, 0.803],   # dark red
#     [0.15, 0.50, 0.15],   # dark green
# ])

TEST_PALETTE_RGB = np.array([
    [0.0, 0.6392, 0.8784],   # red
    [0.0, 0.6588, 0.5608],   # green
    [0.5412, 0.2471, 0.9882],   # orange
    [0.902, 0.0, 0.4039],   # lime green
    [1.0, 0.8196, 0.0],   # dark red
    [0.15, 0.50, 0.15],   # dark green
])  
 
def main():
    # ── Parse arguments ──────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Optimise a colour palette for colour blindness accessibility"
    )
    parser.add_argument("--image",   type=str,   default=None,
                        help="Path to input image")
    parser.add_argument("--colours", type=int,   default=6,
                        help="Number of palette colours (default: 6)")
    parser.add_argument("--cvd",     type=str,   nargs="+",
                        default=["protan", "deutan", "tritan"],
                        choices=["deutan", "protan", "tritan"],
                        help="CVD types to optimise for (default: all three)")
    parser.add_argument("--gens",    type=int,   default=N_GENERATIONS,
                        help=f"GA generations (default: {N_GENERATIONS})")
    parser.add_argument("--pop",     type=int,   default=POP_SIZE,
                        help=f"GA population size (default: {POP_SIZE})")
    parser.add_argument("--tau",     type=float, default=DEFAULT_TAU,
                        help=f"Fidelity threshold in ΔE units (default: {DEFAULT_TAU})")
    parser.add_argument("--runs",    type=int,   default=1,
                        help="Number of independent GA runs (default: 1)")
    parser.add_argument("--save",    action="store_true",
                        help="Save figures to disk instead of displaying")
    args = parser.parse_args()
 
    print("\n" + "=" * 55)
    print("  Colour Palette Optimisation for CVD Accessibility")
    print("=" * 55)
 
    # ── Step 1: Get the palette ───────────────────────────────
    if args.image:
        print(f"\n  Loading image: {args.image}")
        palette_lab, palette_rgb, _, pixel_counts = extract_palette_from_path(
            args.image, k=args.colours
        )
        print(f"\n  Extracted {args.colours}-colour palette:")
        print_palette(palette_lab, palette_rgb, pixel_counts)
    else:
        print("\n  No image provided — using built-in test palette.")
        print("  (Run with --image path/to/image.png to use your own image)\n")
        palette_rgb = TEST_PALETTE_RGB[:args.colours]
        palette_lab = srgb_to_lab(palette_rgb)
        print(f"  Test palette ({len(palette_rgb)} colours):")
        for i, (rgb, lab) in enumerate(zip(palette_rgb, palette_lab)):
            hex_col = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            print(f"    [{i}] {hex_col}  LAB=({lab[0]:.1f}, {lab[1]:.1f}, {lab[2]:.1f})")
 
    # ── Step 2: Run the GA ────────────────────────────────────
    print(f"\n  Optimising for: {', '.join(c.upper() for c in args.cvd)}")
    print(f"  τ = {args.tau} ΔE  |  {args.gens} generations  |  population {args.pop}")
    print(f"  Independent runs: {args.runs}")
    if args.runs > 1:
        print("  (Verbose output suppressed for multi-run mode)\n")
    else:
        print("  This will take 1–3 minutes...\n")
 
    snapshot_dir = "snapshots" if args.save else None
 
    # ── Storage for multi-run statistics ──
    all_scores = []
    all_min_de_per_cvd = {cvd: [] for cvd in args.cvd}
    all_max_drifts = []
    all_mean_drifts = []
    all_histories = []
    all_feasible = []
 
    best_overall_score = -np.inf
    best_overall_palette = None
    best_overall_history = None
 
    for run in range(args.runs):
        if args.runs > 1:
            # Different random seed per run
            np.random.seed(run * 42 + 7)
 
        # Only save snapshots for the first run
        run_snapshot_dir = snapshot_dir if run == 0 else None
 
        optimized_lab, history = run_ga(
            palette_lab,
            cvd_types=args.cvd,
            pop_size=args.pop,
            n_generations=args.gens,
            tau=args.tau,
            verbose=(args.runs == 1),
            snapshot_dir=run_snapshot_dir
        )
 
        # Evaluate this run
        score, details = fitness(
            optimized_lab, palette_lab,
            cvd_types=args.cvd,
            tau=args.tau,
            return_details=True
        )
 
        # Collect statistics
        all_scores.append(score)
        for cvd in args.cvd:
            all_min_de_per_cvd[cvd].append(details["min_de_per_cvd"][cvd])
        all_max_drifts.append(float(np.max(details["drifts"])))
        all_mean_drifts.append(float(np.mean(details["drifts"])))
        all_histories.append(history)
        all_feasible.append(details["feasible"])
 
        # Track global best
        if score > best_overall_score:
            best_overall_score = score
            best_overall_palette = optimized_lab.copy()
            best_overall_history = history
 
        if args.runs > 1:
            print(f"  Run {run + 1:>3d}/{args.runs}  |  "
                  f"fitness = {score:8.3f}  |  "
                  f"max drift = {np.max(details['drifts']):6.2f}  |  "
                  f"feasible = {'Yes' if details['feasible'] else 'No'}")
 
    # ── Step 3: Results ───────────────────────────────────────
    optimized_lab = best_overall_palette
    optimized_rgb = lab_to_srgb(optimized_lab)
 
    if args.runs == 1:
        # Single run — detailed report
        report_result(palette_lab, optimized_lab, cvd_types=args.cvd)
    else:
        # Multi-run — statistical summary
        scores = np.array(all_scores)
        max_d = np.array(all_max_drifts)
        mean_d = np.array(all_mean_drifts)
        n_feasible = sum(all_feasible)
 
        print("\n" + "=" * 70)
        print(f"  STATISTICAL SUMMARY — {args.runs} independent runs  |  τ = {args.tau}")
        print("=" * 70)
 
        print(f"\n  {'Metric':<30} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("  " + "-" * 72)
 
        print(f"  {'Overall fitness':<30} {scores.mean():>8.3f} {np.median(scores):>8.3f} "
              f"{scores.std():>8.3f} {scores.min():>8.3f} {scores.max():>8.3f}")
 
        for cvd in args.cvd:
            vals = np.array(all_min_de_per_cvd[cvd])
            label = f"Min ΔE ({cvd})"
            print(f"  {label:<30} {vals.mean():>8.3f} {np.median(vals):>8.3f} "
                  f"{vals.std():>8.3f} {vals.min():>8.3f} {vals.max():>8.3f}")
 
        print(f"  {'Max per-colour drift':<30} {max_d.mean():>8.3f} {np.median(max_d):>8.3f} "
              f"{max_d.std():>8.3f} {max_d.min():>8.3f} {max_d.max():>8.3f}")
        print(f"  {'Mean per-colour drift':<30} {mean_d.mean():>8.3f} {np.median(mean_d):>8.3f} "
              f"{mean_d.std():>8.3f} {mean_d.min():>8.3f} {mean_d.max():>8.3f}")
 
        print(f"\n  Feasible runs: {n_feasible}/{args.runs} "
              f"({100 * n_feasible / args.runs:.0f}%)")
        print(f"  Best run fitness: {best_overall_score:.3f}")
 
        # Show detailed report for the best run
        print()
        report_result(palette_lab, optimized_lab, cvd_types=args.cvd)
 
    print("  Optimised palette (RGB hex) — best run:")
    for i, rgb in enumerate(optimized_rgb):
        hex_col = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        print(f"    [{i}] {hex_col}")
 
    # ── Step 4: Visualise ─────────────────────────────────────
    palette_save     = "palette_comparison.png" if args.save else None
    convergence_save = "convergence.png"        if args.save else None
 
    print("\n  Showing visualizations...")
 
    # Best palette comparison
    show_palette_comparison(
        palette_lab, optimized_lab,
        cvd_type=args.cvd,
        save_path=palette_save
    )
 
    # Convergence plot — multi-run overlay or single curve
    if args.runs > 1:
        _show_convergence_multi(all_histories, save_path=convergence_save)
    else:
        show_convergence(best_overall_history, save_path=convergence_save)
 
    print("  Done!\n")
 
 
def _show_convergence_multi(all_histories, save_path=None):
    """Plot all convergence curves overlaid with mean ± std."""
    import matplotlib.pyplot as plt
 
    fig, ax = plt.subplots(figsize=(8, 4))
 
    # Each run in light grey
    for history in all_histories:
        ax.plot(history, color="#AAAAAA", linewidth=0.8, alpha=0.5)
 
    # Mean + std band
    max_len = max(len(h) for h in all_histories)
    padded = np.array([
        h + [h[-1]] * (max_len - len(h)) for h in all_histories
    ])
    mean_curve = padded.mean(axis=0)
    std_curve = padded.std(axis=0)
 
    ax.plot(mean_curve, color="#4C72B0", linewidth=2.5, label="Mean")
    ax.fill_between(range(max_len),
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    alpha=0.15, color="#4C72B0", label="±1 Std Dev")
 
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (min ΔE)", fontsize=12)
    ax.set_title(f"GA Convergence — {len(all_histories)} Independent Runs",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
 
 
if __name__ == "__main__":
    main()