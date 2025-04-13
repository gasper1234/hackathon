import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Pisava
plt.rcParams["font.family"] = "Roboto"

# Stil
plt.style.use("default")

# Nastavitve za večje pisave in kontrast
plt.rcParams.update(
    {
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.labelcolor": "#111111",
        "axes.grid": True,
        "grid.color": "black",
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "lines.linewidth": 2,
        "text.color": "#111111",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


# Uporaba palete gist_rainbow
palette = sns.color_palette("gnuplot", 15)
sns.set_palette(palette)


# Primer uporabe palete v drugem file

# from core.plot_style import palette

# plt.plot(
#     t.numpy(),
#     trajectory[:, 4].detach().numpy(),
#     label="PredictedConcentration",
#     color=palette[0],
# )
