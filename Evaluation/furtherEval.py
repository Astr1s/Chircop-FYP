import pickle
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


with open(r"FYP\Mozilla_preprocessed_bug_info.pkl", "rb") as f:
    bug_info = pickle.load(f)

assignee_counts = Counter(info["assignee"] for info in bug_info.values() if info["assignee"].strip())
excluded = {"nobody", "bugzilla", "bugs", "mozilla", "general", "timeless"}
filtered_counts = [count for dev, count in assignee_counts.items() if dev not in excluded]

bug_counts = [count for count in filtered_counts if count < 500]

bugs_fixed_dist = Counter(bug_counts)

x = sorted(bugs_fixed_dist)
y = [bugs_fixed_dist[i] for i in x]

# Line plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='steelblue')
plt.title("Bug Fix Distribution: Number of Developers per Bugs Fixed (<500 Bugs)")
plt.xlabel("Number of Bugs Fixed")
plt.ylabel("Number of Developers")
plt.grid(True)
plt.tight_layout()
plt.show()

one_bug_devs = sum(1 for c in bug_counts if c == 1)
print(f"Number of developers who fixed only one bug: {one_bug_devs}")

def gini_coefficient(x):
    x = np.array(sorted(x))
    n = len(x)
    if n == 0:
        return 0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n

gini = gini_coefficient(bug_counts)
print(f"Gini coefficient: {gini:.4f}")

underrepresented = sum(1 for c in bug_counts if c < 10)
proportion_underrepresented = underrepresented / len(bug_counts)
print(f"Proportion of underrepresented developers (<10 bugs): {proportion_underrepresented:.2%}")
