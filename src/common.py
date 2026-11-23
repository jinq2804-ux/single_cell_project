"""Shared plotting and random seed settings."""
import matplotlib.pyplot as plt

SEED = 42


def save_figure(output, name, fig=None):
    fig = plt.gcf() if fig is None else fig
    fig.savefig(output / "figures" / f"{name}.png", dpi=250,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
