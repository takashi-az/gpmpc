import numpy as np
import time
import matplotlib.pyplot as plt

def k(x,x_):
  return 3*np.exp(-(x-x_)**2/1.5)

def gpr(x_,y_,data):
  t1 = time.time()
  K = np.zeros([len(x_),len(x_)])
  for i in range(len(x_)):
    for j in range(i+1,len(x_)):
      K[i][j] = k(x_[i],x_[j])
      K[j][i] = K[i][j].copy()
  ks = np.zeros([len(x_),len(data)])
  for i in range(len(x_)):
    for j in range(len(data)):
      ks[i][j] = k(x_[i],data[j])
  kss = np.zeros([len(data),len(data)])
  for i in range(len(data)):
    for j in range(len(data)):
      kss[i][j] = k(data[i],data[j])
  print(time.time()-t1)
  var = np.var(data)
  K = K + np.diag([var for i in range(len(x_))])
  Kinv = np.linalg.inv(K)
  mean = ks.T@Kinv@y_.T
  sigma = kss-ks.T@Kinv@ks
  return mean,sigma

def fitc(x_,y_,data):
  ind = 10
  ind_p = np.linspace(-4.5,4.5,10)
  Kfu = np.zeros([len(x_),ind])
  Kff = np.zeros([len(x_),len(x_)])
  for i in range(len(x_)):
    for j in range(i,len(x_)):
      Kff[i][j] = k(x_[i],x_[j])
      Kff[j][i] = Kff[i][j].copy()
    for j in range(len(ind_p)):
      Kfu[i][j] = k(x_[i],ind_p[j])
  Kss = np.zeros([len(data),len(data)])
  Ksu = np.zeros([len(data),ind])
  for i in range(len(data)):
    for j in range(len(data)):
      Kss[i][j] = k(data[i],data[j])
    for j in range(len(ind_p)):
      Ksu[i][j] = k(data[i],ind_p[j])
  Kuu = np.zeros([ind,ind])
  for i in range(len(ind_p)):
    for j in range(len(ind_p)):
      Kuu[i][j] = k(ind_p[i],ind_p[j])
  Kuuinv = np.linalg.inv(Kuu)
  Qss = Ksu@Kuuinv@Ksu.T
  Qff = Kfu@Kuuinv@Kfu.T
  var = np.var(data)
  A = np.diag(Kff-Qff+np.diag([var for i in range(len(x_))]))
  Ainv = np.diag(1./A)
  sig = np.linalg.inv(Kfu.T@Ainv@Kfu+Kuu)
  mean = Ksu@sig@Kfu.T@Ainv@y_.T
  sigma = Kss - Qss + Ksu@sig@Ksu.T
  return mean,sigma

x = np.linspace(-5,5,1000)
ref = np.sin(x)
x1 = np.random.rand(200)*5-5
x2 = np.random.rand(100)*5
x_ = np.concatenate([x1,x2])
y_ = np.sin(x_) + np.random.rand(300)*0.3-0.15
data = np.linspace(-5,5,100)
mean,sigma = gpr(x_,y_,data)
mean_,sigma_ = fitc(x_,y_,data)
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
ax[0].set_xlim(-5,5)
ax[0].set_xlim(-5,5)
ax[1].set_ylim(-1.5,1.5)
ax[1].set_ylim(-1.5,1.5)
ax[0].plot(x,ref)
ax[1].plot(x,ref)
ax[0].scatter(x_,y_,s=7)
ax[1].scatter(x_,y_,s=7)
ax[0].plot(data,mean)
ax[0].fill_between(data, mean+np.diag(sigma), mean-np.diag(sigma), facecolor='orange', alpha=0.5)
ax[1].plot(data,mean_)
ax[1].fill_between(data, mean_+np.diag(sigma_), mean_-np.diag(sigma_), facecolor='orange', alpha=0.5)
ax[0].grid()
ax[1].grid()

plt.show()