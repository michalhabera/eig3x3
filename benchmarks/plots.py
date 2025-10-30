from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

files = sorted(Path(".").glob("*.dat"))
cols = [("J2", (1e-50, 1)),
        ("J3", (1e-65, 1)),
        ("disc", (1e-115, 1)),
        ("eig0", (1e-18, 1)),
        ("eig1", (1e-18, 1)),
        ("eig2", (1e-18, 1))]

for col, ylim in cols:
    if not files:
        print("No .dat files found.")
        break

    fig, axs = plt.subplots(len(files), 1, sharex=True,
                            figsize=(7, 3 * len(files)))
    if len(files) == 1:
        axs = [axs]

    for i, file in enumerate(files):
        try:
            data = np.genfromtxt(file, names=True)
            delta = data["delta"]
            err = data[f"{col}_errors_c"]
            conds = data[f"{col}_conds"]

            # Sort by delta
            order = np.argsort(delta)

            # Check for missed values
            safety_factor = 10
            failed = err > safety_factor * conds
            if np.any(failed):
                print(
                    f"{col}/{file.name}: {100 * np.sum(failed) / err.size}% failed values")

            # Plot
            ax = axs[i]
            ax.set_title(file.stem, fontsize="small")
            ax.loglog(delta[order], err[order], "k+", label="C")
            ax.loglog(delta[order], conds[order],
                      "r-", linewidth=2, label="stability bound")
            ax.set_ylabel("forward error")
            ax.set_xlabel(r"$\delta$")
            ax.grid(True, alpha=0.5)
            ax.legend(fontsize="small")
            # ax.set_ylim(ylim)

        except Exception as e:
            print(f"Skipping {file.name}: {e}")

    fig.tight_layout()
    fig.savefig(f"{col}.png", dpi=200)
    plt.close()
