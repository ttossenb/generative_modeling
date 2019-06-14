import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions

from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse


k = 100
d = 2


def vis_contours(sess, components):
    assert d == 2 # sry
    n = 100
    x = np.linspace(-4., +4., n)
    y = np.linspace(-4., +4., n)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    for component in components:
        densities = -sess.run(component.log_prob(XX))
        Z = np.array(densities).reshape((n, n))
        CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
    plt.show()


def vis_scales(sess, means, scales):
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    means_np, scales_np = sess.run([means, scales])
    ells = [Ellipse(xy=means_np[i],
                        width =2 * scales_np[i, 0],
                        height=2 * scales_np[i, 1])
                for i in range(k)]
    for i in range(k):
        ax.add_artist(ells[i])
        ells[i].set_clip_box(ax.bbox)
        ells[i].set_alpha(0.5)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.show()
    plt.clf()
    plt.close()
    plt.scatter(means_np[:, 0], scales_np[:, 0], c='blue')
    plt.scatter(means_np[:, 1], scales_np[:, 1], c='red')
    plt.show()


def grid(r, n):
    x = np.linspace(-r, +r, n)
    y = np.linspace(-r, +r, n)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    return XX


with tf.Session() as sess:
    sqrt_k = int(np.sqrt(k)) ; assert sqrt_k * sqrt_k == k
    means = tf.Variable(grid(3, sqrt_k).astype(np.float32))
    means = tf.Variable(tf.random.normal(k, d))
    scales =  tf.Variable(tf.ones(k, d))

    components = [tfd.MultivariateNormalDiag(loc=means[i], scale_diag=scales[i]) for i in range(k)]

    dist = tfd.Mixture(
        cat=tfd.Categorical(probs=[1.0/k for _ in range(k)]),
        components=components
    )

    init = tf.initialize_all_variables()
    sess.run(init)

    prior = tfd.MultivariateNormalDiag(loc=np.zeros((d, ), dtype=np.float32), scale_diag=np.ones((d, ), dtype=np.float32))
    n = 10000
    samples = dist.sample(n)
    kl = tf.reduce_mean(dist.log_prob(samples) - prior.log_prob(samples))
    # kl = dist.kl_divergence(prior) # unimplemented for mixtures.

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_step = optimizer.minimize(kl, var_list=scales)

    init = tf.initialize_all_variables()
    sess.run(init)

    for i in range(2500):
        sess.run(train_step)
        if i % 100 == 0:
            print(sess.run(kl))

    # vis_contours(sess, components)
    vis_scales(sess, means, scales)
