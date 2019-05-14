import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Ellipse
import scipy.stats

import os

import model_IO
import params
import data
import vis


def cdf(x):
    return scipy.stats.norm.cdf(x, loc=0, scale=1)

def icdf(x):
    return scipy.stats.norm.isf(1 - x, loc=0, scale=1)

n = 100

xs = []
ys = []

"""
for r in np.linspace(0, 2, 10):
    for i in range(n+1):
        x = np.cos(i*2*np.pi/n) * r
        y = np.sin(i*2*np.pi/n) * r
        xs.append(cdf(x))
        ys.append(cdf(y))
        plt.plot(xs, ys)

plt.show()
"""

m = 40
n = m ** 2

for u in np.linspace(0, 1, m+2)[1:-1]:
    for v in np.linspace(0, 1, m+2)[1:-1]:
        xs.append(icdf(u))
        ys.append(icdf(v))

#print(len(xs))
#print(len(ys))

"""
latentpoints = np.zeros((len(xs), 2))

for i in range(len(xs)):
    latentpoints[i][0] = xs[i]
    latentpoints[i][1] = ys[i]
"""

#latentpoints = np.array(zip(xs, ys))
latentpoints = np.array([xs, ys]).T

#print(latentpoints)

args = params.getArgs()
modelDict = model_IO.load_autoencoder(args)
encoder = modelDict.encoder
generator = modelDict.generator

data_object = data.load(args.dataset, shape=args.shape, color=args.color)
((x_train, y_train), (x_test, y_test)) = data_object.get_data(args.trainSize, args.testSize)
args.original_shape = x_train.shape[1:]
args.original_size = np.prod(args.original_shape)

latims = x_train[:1000]
labels = y_train[:1000]

z_sampled, z_mean, z_logvar = encoder.predict(latims, args.batch_size)
images = generator.predict(latentpoints, args.batch_size)

inum = len(latims)

ells = [Ellipse(xy = z_mean[i],
                width = 2 * np.exp(z_logvar[i][0]),
                height = 2 * np.exp(z_logvar[i][1]))
        for i in range(inum)]     
    
#print(images.shape)

#vis.plotImages(images, m, m, "grid")

"""
for p, img in zip(latentpoints, images):
    plt.imshow(img[:,:,0], extent = [p[0]-0.1, p[0]+0.1, p[1]-0.1, p[1]+0.1])


plt.show()
"""

def getImage(i, zoom=0.05):
    return OffsetImage(images[i][:,:,0], zoom=zoom)

#fig = plt.figure(10, 10)

fig, ax = plt.subplots(subplot_kw = {'aspect' : 'equal'})
ax.scatter(xs, ys, s = None, c = "white")

for i in range(n):
    ab = AnnotationBbox(getImage(i), (latentpoints[i][0], latentpoints[i][1]), frameon=False)
    ax.add_artist(ab)

blu = []
g = [] 
r = []
c = []
m = [] 
y = []
bla = []
o = []
t = []
br = []

for i in range(inum):
        ax.add_artist(ells[i])
        ells[i].set_clip_box(ax.bbox)
        ells[i].set_alpha(0.5)
        if labels[i] == 0:     
                ells[i].set_facecolor('blue')
                blu.append(ells[i])
        elif labels[i] == 1:
                ells[i].set_facecolor('green')
                g.append(ells[i])     
        elif labels[i] == 2:
                ells[i].set_facecolor('red')
                r.append(ells[i])
        elif labels[i] == 3:
                ells[i].set_facecolor('cyan')
                c.append(ells[i])
        elif labels[i] == 4:
                ells[i].set_facecolor('magenta')
                m.append(ells[i])
        elif labels[i] == 5:
                ells[i].set_facecolor('yellow')
                y.append(ells[i])
        elif labels[i] == 6:
                ells[i].set_facecolor('black')
                bla.append(ells[i])
        elif labels[i] == 7:
                ells[i].set_facecolor('orange')
                o.append(ells[i])
        elif labels[i] == 8:
                ells[i].set_facecolor('teal')
                t.append(ells[i])
        else:
                ells[i].set_facecolor('brown')
                br.append(ells[i])
        
ax.legend((blu[0], g[0], r[0], c[0], m[0], y[0], bla[0], o[0], t[0], br[0]), 
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), loc="best")

plt.savefig("gridnumbers.png", dpi = 1000)



