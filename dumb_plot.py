import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

a = np.array(list(map(float, sys.stdin.readlines())))
print(a.shape)
plt.figure(figsize=(20, 10))
plt.plot(a)
plt.savefig(sys.argv[1])
