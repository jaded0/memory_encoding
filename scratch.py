import numpy as np

def map_to_positive(value, shift=1, scale=1e2):
    # Shift and scale the input value
    shifted_value = value + shift
    # Apply the logistic function with scaling
    positive_value = scale / (1 + np.exp(-shifted_value))
    return positive_value

# Example usage
values = [-20, -2, -1, 0, 1, 2]
mapped_values = [map_to_positive(v) for v in values]
print(mapped_values)
