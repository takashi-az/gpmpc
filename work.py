import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
ax.plot([0,1,2,3,4,5],[4.23,1.07,1.38,0.92,0.87,0.47])
ax.set_xlabel('episode')
ax.set_ylabel('RMSE')
ax.grid()
plt.show()