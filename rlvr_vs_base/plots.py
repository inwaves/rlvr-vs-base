"""pass@k curve plotting."""

from pathlib import Path


def plot_passk(payload: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = payload["ks"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key, color, label in (("base", "#1f77b4", "Base"), ("rl", "#d62728", "RL Zero")):
        d = payload[key]
        ax.plot(ks, d["mean"], marker="o", ms=3.5, color=color, label=label)
        ax.fill_between(ks, d["lo"], d["hi"], color=color, alpha=0.15, lw=0)
    if payload.get("crossover_k"):
        ax.axvline(payload["crossover_k"], color="grey", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=7)
    ax.set_xlabel("Number of samples k")
    ax.set_ylabel("Coverage (pass@k)")
    ax.set_title(f"{payload['benchmark']} — {payload['problems']} problems")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
