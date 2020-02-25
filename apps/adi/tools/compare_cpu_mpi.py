#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

mpi_array = np.genfromtxt('./data/mpi-i-0.csv', delimiter=' ')
cpu_array = np.genfromtxt('./data/cpu-i-0.csv', delimiter=' ')

print(mpi_array.shape)
print(cpu_array.shape)

diff = np.abs(mpi_array - cpu_array)

print("Max difference: " + str(np.max(diff)))
print("Min difference: " + str(np.min(diff)))
print("Median difference: " + str(np.median(diff)))
print("Sum difference: " + str(np.sum(diff)))
print("Avg difference: " + str(np.sum(diff) / diff.size))

sqdiff = np.square(diff)

print("MSE: " + str(np.sum(sqdiff) / sqdiff.size))
print("SE: " + str(np.sum(sqdiff)))

# Plot heat map of one layer
plt.imshow(mpi_array[63 * 128 : 64 * 128, :], cmap='hot')
plt.show()

# Plot error points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

nz = ny = nx = 64
t = np.max(diff) / 2

x_coords = []
y_coords = []
z_coords = []

for z in range(nz):
  for y in range(ny):
    if max(diff[z * ny + y]) > t:
      for x in range(nx):
        if diff[z * ny + y][x] > t:
          x_coords.append(x)
          y_coords.append(y)
          z_coords.append(z)
          
ax.scatter(x_coords, y_coords, z_coords)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
