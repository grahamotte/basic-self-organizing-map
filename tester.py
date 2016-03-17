import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from network import NG,SOM,DSOM
from distribution import uniform, normal, ring

n = 8
epochs = 20000
N = 10000

np.random.seed(123)
samples = ring(n=N, radius=(0.25,0.5) )

print 'Self-Organizing Map'
np.random.seed(123)
som = SOM((n,n,2))
som.learn(samples,epochs)

print 'Dynamic Self-Organizing Map'
np.random.seed(123)
dsom = DSOM((n,n,2), elasticity=1.75)
dsom.learn(samples,epochs)

fig = plt.figure(figsize=(21,8))
fig.patch.set_alpha(0.0)

axes = fig.add_subplot(1,3,2)
som.plot(axes)
axes = fig.add_subplot(1,3,3)
dsom.plot(axes)
fig.savefig('out.png',dpi=150)
