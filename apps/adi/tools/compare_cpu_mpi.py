#!/usr/bin/env python3

import numpy as np

mpi_array = np.genfromtxt('../build/mpi-n128-i1.csv', delimiter=' ')
cpu_array = np.genfromtxt('../build/cpu-n128-i1.csv', delimiter=' ')

difference = np.abs(mpi_array - cpu_array)

print("Max difference: " + str(np.max(difference)))
print("Min difference: " + str(np.min(difference)))
print("Median difference: " + str(np.median(difference)))
print("Sum difference: " + str(np.sum(difference)))
print("Avg difference: " + str(np.sum(difference) / difference.size))
