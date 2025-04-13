import pandas as pd
import plot_style
import matplotlib.pyplot as plt

# Load data from CSV
file_path = "/home/david/Hackathon/DownstreamProcessIntensification/results_multiple_optimizations.csv"
data = pd.read_csv(
    file_path,
    header=0,
    names=["final_yield_optimized", "final_yield_constant", "purity"],
)
# print(data["final_yield_constant"])
# Calculate improvement in yield in percentage
data["yield_improvement"] = (
    (data["final_yield_optimized"] - data["final_yield_constant"])
    / data["final_yield_constant"]
) * 100

# Group by purity and calculate the average yield improvement
average_yield_improvement = data.groupby("purity")["yield_improvement"].mean()

# Plot
plt.figure(figsize=(8, 6))
plt.plot(
    average_yield_improvement.index,
    average_yield_improvement.values,
    marker="o",
    linestyle="-",
    color="b",
    label="Average Yield Improvement",
)

# Labels and title
plt.xlabel("Purity")
plt.ylabel("Average Yield Improvement (%)")
plt.title("Average Yield Improvement vs Purity")
plt.grid(True)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()
