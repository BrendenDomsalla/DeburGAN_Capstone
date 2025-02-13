import matplotlib.pyplot as plt
import random

# Constants
S = 50  # Side length of each square
O = 10  # Overlap amount

# Compute step size
d = S - O  # Distance between top-left corners of adjacent squares

# Random grid size
G_x = random.randint(5, 10)  # Random number of squares horizontally
G_y = random.randint(5, 10)  # Random number of squares vertically

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, G_x * d + O)
ax.set_ylim(0, G_y * d + O)
ax.set_aspect('equal')

# Draw squares
for i in range(G_x):
    for j in range(G_y):
        x = i * d
        y = j * d
        square = plt.Rectangle(
            (x, y), S, S, edgecolor='black', facecolor='lightblue', alpha=0.6)
        ax.add_patch(square)

# Hide axes
ax.set_xticks([])
ax.set_yticks([])
ax.invert_yaxis()  # Invert to match normal (0,0) top-left coordinate system

plt.show()
