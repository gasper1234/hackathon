import pandas as pd

import plot_style

import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv("results_yield_time_purity.csv")

# Group data by purity
grouped = data.groupby("purity")

# Plot yield vs time for each purity
plt.figure(figsize=(8, 6))
for purity, group in grouped:
    # order the data by time
    group = group.sort_values("time")

    plt.plot(group["time"], group["yield"], marker="o", label=f"Purity: {purity}")

# Add labels, legend, and title
plt.xlabel("Time")
plt.ylabel("Yield")
plt.title("Yield vs Time for Different Purity Levels")
plt.legend()
plt.grid(True)

# Show the plot
# plt.tight_layout()
plt.show()
plt.savefig("yield_vs_time.png")
