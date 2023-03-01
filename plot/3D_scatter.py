import matplotlib.pyplot as plt
from plot.np_data_generator import get_XYZ

X, Y, Z = get_XYZ()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z)
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('rev lift')
plt.show()
