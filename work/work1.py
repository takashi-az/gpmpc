import time
import numpy as np

t1 = time.time()
for i in range(3):
  a = np.random.randn(300,300)
  for j in range(100):
    u,_,_ = np.linalg.svd(a)
print(time.time()-t1)