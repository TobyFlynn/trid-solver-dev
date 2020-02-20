#!/usr/bin/env python3

import numpy as np

#cpu_array = np.genfromtxt('../build/cpu.csv', delimiter=' ', max_rows=1000)
#mpi_array = np.genfromtxt('../build/mpi.csv', delimiter=' ', max_rows=1000)
mpi_1 = np.genfromtxt('../build/mpi-1-y.csv', delimiter=' ')

print("MPI 1: " + str(len(mpi_1)) + "x" + str(len(mpi_1[0])))

mpi_2_1 = np.genfromtxt('../build/mpi-2-y.csv', delimiter=' ', skip_footer=128)
mpi_2_2 = np.genfromtxt('../build/mpi-2-y.csv', delimiter=' ', skip_header=128)

mpi_2 = np.concatenate((mpi_2_1, mpi_2_2), axis=0)
print("MPI 2: " + str(len(mpi_2)) + "x" + str(len(mpi_2[0])))

mpi_3_1 = np.genfromtxt('../build/mpi-3-y.csv', delimiter=' ', skip_footer=170)
mpi_3_2 = np.genfromtxt('../build/mpi-3-y.csv', delimiter=' ', skip_header=86, skip_footer=85)
mpi_3_3 = np.genfromtxt('../build/mpi-3-y.csv', delimiter=' ', skip_header=171)

mpi_3 = np.concatenate((mpi_3_1, mpi_3_2, mpi_3_3), axis=0)
print("MPI 3: " + str(len(mpi_3)) + "x" + str(len(mpi_3[0])))

mpi_4_1 = np.genfromtxt('../build/mpi-4-y.csv', delimiter=' ', skip_footer=64*3)
mpi_4_2 = np.genfromtxt('../build/mpi-4-y.csv', delimiter=' ', skip_header=64, skip_footer=64*2)
mpi_4_3 = np.genfromtxt('../build/mpi-4-y.csv', delimiter=' ', skip_header=64*2, skip_footer=64)
mpi_4_4 = np.genfromtxt('../build/mpi-4-y.csv', delimiter=' ', skip_header=64*3)

mpi_4 = np.concatenate((mpi_4_1, mpi_4_2, mpi_4_3, mpi_4_4), axis=0)
print("MPI 4: " + str(len(mpi_4)) + "x" + str(len(mpi_4[0])))

mpi_5_1 = np.genfromtxt('../build/mpi-5-y.csv', delimiter=' ', skip_footer=204)
mpi_5_2 = np.genfromtxt('../build/mpi-5-y.csv', delimiter=' ', skip_header=52, skip_footer=51*3)
mpi_5_3 = np.genfromtxt('../build/mpi-5-y.csv', delimiter=' ', skip_header=103, skip_footer=51*2)
mpi_5_4 = np.genfromtxt('../build/mpi-5-y.csv', delimiter=' ', skip_header=154, skip_footer=51)
mpi_5_5 = np.genfromtxt('../build/mpi-5-y.csv', delimiter=' ', skip_header=205)

mpi_5 = np.concatenate((mpi_5_1, mpi_5_2, mpi_5_3, mpi_5_4, mpi_5_5), axis=0)
print("MPI 5: " + str(len(mpi_5)) + "x" + str(len(mpi_5[0])))

#mpi_2 = np.genfromtxt('../build/mpi-2.csv', delimiter=' ', max_rows=1000)
#mpi_3 = np.genfromtxt('../build/mpi-3.csv', delimiter=' ', max_rows=1000)
#mpi_4 = np.genfromtxt('../build/mpi-4.csv', delimiter=' ', max_rows=1000)
#mpi_5 = np.genfromtxt('../build/mpi-5.csv', delimiter=' ', max_rows=1000)

diff_1 = np.abs(mpi_2 - mpi_1)
diff_2 = np.abs(mpi_3 - mpi_1)
diff_3 = np.abs(mpi_4 - mpi_1)
diff_4 = np.abs(mpi_5 - mpi_1)

print("Max diff_1: " + str(np.max(diff_1)))
print("Sum diff_1: " + str(np.sum(diff_1)))

print("Max diff_2: " + str(np.max(diff_2)))
print("Sum diff_2: " + str(np.sum(diff_2)))

print("Max diff_3: " + str(np.max(diff_3)))
print("Sum diff_3: " + str(np.sum(diff_3)))

print("Max diff_4: " + str(np.max(diff_4)))
print("Sum diff_4: " + str(np.sum(diff_4)))

print("####### diff_1 ########")

for i in range(len(diff_1)):
  if max(diff_1[i]) > 0.0:
    print("Row " + str(i) + ":")
    for j in range(len(diff_1[i])):
      print("%.1g " % diff_1[i][j], end='')
    print()

print("####### diff_2 ########")

for i in range(len(diff_2)):
  if max(diff_2[i]) > 0.0:
    print("Row " + str(i) + ":")
    for j in range(len(diff_2[i])):
      print("%.1g " % diff_2[i][j], end='')
    print()
    
print("####### diff_3 ########")

for i in range(len(diff_3)):
  if max(diff_3[i]) > 0.0:
    print("Row " + str(i) + ":")
    for j in range(len(diff_3[i])):
      print("%.1g " % diff_3[i][j], end='')
    print()
    
print("####### diff_4 ########")

for i in range(len(diff_4)):
  if max(diff_4[i]) > 0.0:
    print("Row " + str(i) + ":")
    for j in range(len(diff_4[i])):
      print("%.1g " % diff_4[i][j], end='')
    print()
