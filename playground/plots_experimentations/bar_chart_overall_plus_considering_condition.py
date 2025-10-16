import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Example DataFrame - replace with your actual data
df = pd.DataFrame(
    {
        "num_of_additions": [5, 12, 8, 25, 30, 15, 22, 7, 18, 35],
        "condition1": [True, False, True, True, False, True, False, True, True, False],
    }
)

# Define ranges for grouping
bins = [0, 10, 20, 30, 40]
labels = ["0-10", "11-20", "21-30", "31-40"]

# Create a new column for the range
df["range"] = pd.cut(df["num_of_additions"], bins=bins, labels=labels, right=False)

# Calculate sums for all data and for condition1=True
all_sums = df.groupby("range", observed=True)["num_of_additions"].sum()
condition_sums = (
    df[df["condition1"]].groupby("range", observed=True)["num_of_additions"].sum()
)

# Reindex to ensure all ranges are present (fill missing with 0)
all_sums = all_sums.reindex(labels, fill_value=0)
condition_sums = condition_sums.reindex(labels, fill_value=0)

# Create the bar chart
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width / 2, all_sums, width, label="All Data", alpha=0.8)
bars2 = ax.bar(x + width / 2, condition_sums, width, label="Condition1=True", alpha=0.8)

# Customize the chart
ax.set_xlabel("Ranges", fontsize=12)
ax.set_ylabel("Sum of num_of_additions", fontsize=12)
ax.set_title("Sum of Additions by Range", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

plt.tight_layout()
plt.show()
