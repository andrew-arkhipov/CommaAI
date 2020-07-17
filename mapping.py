import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

Rt = np.array([[9.99991076e-01, 7.11603037e-04, -4.16431458e-03, -6.38135578e-03],
                [-7.19138763e-04, 9.99998106e-01, -1.80837836e-03, -1.18687754e-01],
                [4.16301985e-03, 1.81135694e-03, 9.99989694e-01, 9.92911122e-01],
                [0, 0, 0, 1]], dtype=np.float32)
K = np.array(([140, 0, 320, 0], [0, 140, 240, 0], [0, 0, 1, 0]))
M = np.linalg.pinv(np.matmul(K, Rt))

fig = plt.figure()
ax = plt.axes(projection='3d')

x = []
y = []
z = []

for i in range(0, 640):
    for j in range(240, 480):
        coord = np.array([i, j, 1])
        res = np.matmul(M, coord)
        xc, yc, zc = res[0], res[1], res[2]
        d = -Rt[1, 3]/(yc - Rt[1, 3])
        # print(d)
        x.append((xc - Rt[0, 3])*d + Rt[0, 3])
        y.append(0)
        z.append((zc - Rt[2, 3])*d + Rt[2, 3])


ax.scatter3D(x, z, y)
plt.show()