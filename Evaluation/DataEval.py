from collections import Counter
import pickle
import numpy as np

# Load the data
with open("Eclipse_preprocessed_bug_info.pkl", "rb") as f:
    bug_info = pickle.load(f)

fake_users = {"nobody", "bugzilla", "bugs", "mozilla", "timeless", 
              "general", "justdave", "postmaster", "admin",'webmaster@eclipse.org','rap-inbox@eclipse.org'}

assignees = [a for a in (info["assignee"].strip().lower() for info in bug_info.values()) if a and a not in fake_users]
counter = Counter(assignees)


print("Top 5 real developers with most fixed bugs:")
for i, (dev, count) in enumerate(counter.most_common(5), 1):
    print(f"{i}. {dev}: {count} bugs")

counts = list(counter.values())
avg = np.mean(counts)
below_avg = sum(1 for c in counts if c < avg)

print(f"\nAverage number of fixed bugs per developer: {avg:.2f}")
print(f"Number of developers below average: {below_avg} out of {len(counts)}")
