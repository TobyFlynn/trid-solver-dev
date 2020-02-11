#!/usr/bin/env python3

import numpy as np

mpi_array = np.genfromtxt('../build/mpi.csv', delimiter=' ')
cpu_array = np.genfromtxt('../build/cpu.csv', delimiter=' ')

difference = np.abs(mpi_array - cpu_array)

print("Max difference: " + str(np.max(difference)))
print("Sum difference: " + str(np.sum(difference)))
