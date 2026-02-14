# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 01:17:20 2024

@author: ossam
"""

import numpy as np

numbers = []
# Load data from text file
with open('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7/v7_recall_P.txt', 'r') as file:
    for line in file:
        line_numbers = [float(num) for num in line.split()]
        numbers.extend(line_numbers)

data_flat = np.array(numbers)
# Flatten the array to convert it into a one-dimensional array
# data_flat = data.flatten()

# Print the flattened array
print(data_flat)
np.savetxt('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7/v7_recall_P_1row.txt', data_flat, fmt='%.8f', newline=' ')

# import numpy as np

# # Load data from text file
# data = np.loadtxt('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7/v7_presicion_U.txt', dtype=float)

# # Flatten the array to convert it into a one-dimensional array
# data_flat = data.flatten()

# # Save the flattened data to a text file with one row
# np.savetxt('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7/v7_presicion_U_1row.txt', data_flat, fmt='%.8f', newline=' ')
