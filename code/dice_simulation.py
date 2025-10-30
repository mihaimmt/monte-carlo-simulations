# dice_simulation_simple.py
import random
import os
import matplotlib.pyplot as plt

# -------------------------------
# Ensure results folder exists
# -------------------------------
os.makedirs("results", exist_ok=True)

# -------------------------------
# Parameters
# -------------------------------
N = 10000  # number of dice rolls
faces = 6  # sides of the dice

# -------------------------------
# Simulate dice rolls
# -------------------------------
rolls = []
for _ in range(N):
    roll = random.randint(1, faces)  # generates integer 1-6
    rolls.append(roll)

# -------------------------------
# Count occurrences
# -------------------------------
counts = [0] * faces  # list to store counts for faces 1-6

for roll in rolls:
    counts[roll - 1] += 1  # increment count for that face

# -------------------------------
# Calculate empirical probabilities
# -------------------------------
freqs = [count / N for count in counts]

# -------------------------------
# Print results
# -------------------------------
print("Counts per face:", counts)
print("Empirical probabilities:", freqs)
print("Expected probability per face:", 1 / faces)

# -------------------------------
# Plot histogram
# -------------------------------
plt.bar(range(1, faces + 1), freqs) # x values on histogram are 1-6, y values are frequencies
plt.xlabel("Face") # x-axis label
plt.ylabel("Frequency") # y-axis label
plt.title(f"Dice frequencies (n={N})") # title of histogram
plt.savefig("results/dice_freqs.png", dpi=150) # save figure to results folder 
plt.show() # display histogram
