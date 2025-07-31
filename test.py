import numpy as np

array = np.array([[1, 7, 3], [4, 4, 6], [7, 8, 0]])
r = np.where(array == 0)
print(r[0][0], r[1][0])  # Output the row and column index of the zero element