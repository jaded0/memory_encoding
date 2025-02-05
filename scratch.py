import numpy as np
import matplotlib.pyplot as plt

d_model = 5         # Embedding dimension
num_positions = 10   # Number of sequence positions to visualize

# Prepare an array to hold the positional encodings:
# Shape: (num_positions, d_model)
pos_enc = np.zeros((num_positions, d_model))

# Compute the sine/cosine encoding for each position and dimension
for pos in range(num_positions):
    for i in range(d_model // 2):
        # Even dimension: sine
        pos_enc[pos, 2*i] = np.sin(
            pos / (10000 ** ((2*i) / d_model))
        )
        # Odd dimension: cosine
        pos_enc[pos, 2*i + 1] = np.cos(
            pos / (10000 ** ((2*i) / d_model))
        )

# Plot a scatter for each dimension, coloring by dimension index
colors = plt.cm.tab20(np.linspace(0, 1, d_model))  # 20 distinct colors
plt.figure(figsize=(8, 5))
for dim in range(d_model):
    plt.scatter(
        x=np.arange(num_positions),
        y=pos_enc[:, dim],
        color=colors[dim],
        label=f'dim {dim}',
        s=50
    )

plt.xlabel('Position')
plt.ylabel('Positional Encoding Value')
plt.title(f'Positional Encodings (d_model={d_model}, positions={num_positions})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
