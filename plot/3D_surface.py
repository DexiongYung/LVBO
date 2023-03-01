from matplotlib import cm
import matplotlib.pyplot as plt
from plot.np_data_generator import get_XYZ

X, Y, Z = get_XYZ()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)
ax.set_xlabel("theta0")
ax.set_ylabel("theta1")
ax.set_zlabel("rev lift")
fig.colorbar(surf)
plt.show()
