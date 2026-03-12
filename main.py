# A list of errors
numbers = [10, 20, 30]

# To get the gradient (2 * error), we must loop
multiples = []
for e in numbers:
    multiples.append(2 * e)

print(multiples) # Output: [20, 40, 60]


import numpy as np

# Convert list to a NumPy Vector (1D Array)
numbers_vec = np.array([10, 20, 30])

# No loop needed! NumPy multiplies every element instantly.
multiples = 2 * numbers_vec

print(multiples) # Output: [20, 40, 60]