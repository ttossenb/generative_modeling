from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Input, Lambda
from keras import backend as K
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import networkx as nx
import sys
import time

from util import *
import model_IO
import loss
import vis
import samplers
import callbacks


from networks import dense, conv

import NAT_graph


# TODO move to vis.py
def plot_matching(latents, nats, matching, filename):
    n = len(latents)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(latents[:, 0], latents[:, 1], c='red')
    ax.scatter(nats[:, 0], nats[:, 1], c='blue')
    for latent_index, nat_index in enumerate(matching):
        if nat_index is not None:
            xs = [latents[latent_index, 0], nats[nat_index, 0]]
            ys = [latents[latent_index, 1], nats[nat_index, 1]]
            l = Line2D(xs, ys)
            ax.add_line(l)
    ax.set_xlim(-2, +2)
    ax.set_ylim(-2, +2)
    plt.title(filename)
    plt.savefig(filename)
    plt.close()


def pairwiseSquaredDistances(clients, servers):
    cL2S = np.sum(clients ** 2, axis=-1)
    sL2S = np.sum(servers ** 2, axis=-1)
    cL2SM = np.tile(cL2S, (len(servers), 1))
    sL2SM = np.tile(sL2S, (len(clients), 1))
    squaredDistances = cL2SM + sL2SM.T - 2.0 * servers.dot(clients.T)
    return squaredDistances.T


def weight_of_matching(matching, latentPositions, natPositions):
    w = 0.0
    n = len(latentPositions)
    for client, server in enumerate(matching):
        if server is not None:
            assert latentPositions[client, 0] != np.nan
            w += np.linalg.norm(latentPositions[client] - natPositions[server])
    return w


# builds the whole matrix from scratch, disregards dirties, bipartite, annoyIndex.
def updateBipartiteGraphFromScratch(latentPositions, natPositions):
    distances = np.sqrt(pairwiseSquaredDistances(latentPositions, natPositions))
    n = distances.shape[0]
    bipartite = nx.Graph()
    bipartite.add_nodes_from(range(n), bipartite=0)
    bipartite.add_nodes_from(range(n, 2*n), bipartite=1)
    for i in range(n):
        for j in range(n):
            bipartite.add_edge(i, n+j, weight=-distances[i, j], near=False)
    matching = nx.algorithms.matching.max_weight_matching(bipartite, maxcardinality=True, weight='weight')
    m2 = [None for _ in range(n)]
    for a, b in matching:
        if a >= n:
            b, a = a, b
        latent_index = a
        nat_index = b - n
        m2[latent_index] = nat_index
    return m2


# TODO refactor, move to its own py
def updateBipartiteGraph(dirties, latentPositions, natPositions, bipartite, annoyIndex):
    n_nearest = 5 # 0
    n_random = 5 # len(natPositions)

    n = len(natPositions)
    totalChange = 0.0
    changed = 0

    distances = np.sqrt(pairwiseSquaredDistances(latentPositions, natPositions))

    for i in dirties:
        latent = latentPositions[i]

        
        closeIndices = np.argsort(distances[i])[:n_nearest]
        closeDistances = []
        for j in closeIndices:
            closeDistances.append(distances[i][j]) 

        '''
        closeIndices, closeDistances = annoyIndex.get_nns_by_vector(
            latent, n_nearest, include_distances=True)
        '''

        oldNeighbors = None
        if i in bipartite:
            oldNeighbors = set([nei for nei, datadict in bipartite.adj[i].items() if datadict['near']])
            bipartite.remove_node(i)
        bipartite.add_node(i, bipartite=0)
        newNeighbors = {closeNatIndex + n for closeNatIndex in closeIndices}
        if oldNeighbors is not None and len(oldNeighbors) > 0:
            inters = len(newNeighbors.intersection(oldNeighbors))
            jaccardDistance = 1 - inters / (2 * len(oldNeighbors) - inters)
            totalChange += jaccardDistance
            changed += 1
        for j in range(len(closeIndices)):
            closeNatIndex = closeIndices[j]
            dist = closeDistances[j]
            bipartite.add_edge(i, closeNatIndex + n, weight= dist * (-1), near=True) # minus because maximum weight
        randomIndices = np.random.choice(n, size=n_random, replace=False)
        for natIndex in sorted(randomIndices):
            dist = np.linalg.norm(natPositions[natIndex] - latent)
            bipartite.add_edge(i, natIndex + n, weight= dist * (-1), near=False) # minus because maximum weight
    print("Average change", totalChange / (changed+0.01), changed, len(latentPositions))

    print("nodes", bipartite.number_of_nodes(), "edges", bipartite.number_of_edges())
    matching = nx.algorithms.matching.max_weight_matching(bipartite, maxcardinality=True, weight='weight')
    print("matching", len(matching))
    # TODO print the ratio of near nodes in the matching
    m2 = [None for _ in range(n)]
    for a, b in matching:
        if a >= n:
            b, a = a, b
        latent_index = a
        nat_index = b - n
        m2[latent_index] = nat_index

    sys.stdout.flush()
    return m2


def hashSample(sample):
    return hash(tuple(sample.flatten()))


def run(args, data):
    # (x_train, x_test) = data

    ((x_train, y_train), (x_test, y_test)) = data
    images = x_train[:1000].copy()
    labels = y_train[:1000].copy()

    print(x_test.shape)

    sampler = samplers.sampler_factory(args)

    models, loss_features = build_models(args)
    assert set(("ae", "encoder", "generator")) <= set(
        models.keys()), models.keys()

    print("Encoder architecture:")
    print_model(models.encoder)
    print("Generator architecture:")
    print_model(models.generator)

    # get losses
    loss_names = sorted(set(args.loss_encoder + args.loss_generator))
    losses_without_nat = loss.loss_factory(
        loss_names, args, loss_features, combine_with_weights=True)

    nat_loss_weight_variable = K.variable(0.0)
    nat_loss_tensor = nat_loss_weight_variable * K.mean(K.square(loss_features.z_mean - loss_features.z_nat))
    def nat_loss(x, x_decoded):
        return nat_loss_tensor
    def losses_with_nat(x, x_decoded):
        return losses_without_nat(x, x_decoded) + nat_loss_tensor
    losses = losses_with_nat

    metric_names = sorted(set(args.metrics + tuple(loss_names)))
    metrics = loss.loss_factory(
        metric_names, args, loss_features, combine_with_weights=False)
    metrics += [nat_loss]

    # get optimizer
    if args.optimizer == "rmsprop":
        optimizer = RMSprop(lr=args.lr, clipvalue=1.0)
    elif args.optimizer == "adam":
        optimizer = Adam(lr=args.lr, clipvalue=1.0)
    elif args.optimizer == "sgd":
        optimizer = SGD(lr=args.lr, clipvalue=1.0)
    else:
        assert False, "Unknown optimizer %s" % args.optimizer

    # compile autoencoder
    models.ae.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    imageDisplayCallback = callbacks.ImageDisplayCallback(
        x_train, x_test, args, models, sampler)

    iters_in_epoch = x_train.shape[0] // args.batch_size

    n = len(x_train)
    print("creating bipartite graph and annoy index.")
    natPositions, bipartite, annoyIndex = NAT_graph.buildServers(
        n, args.latent_dim)
    print("done.")

    latentPositions = np.zeros_like(natPositions)
    latentPositions.fill(np.nan)

    # TODO rewrite with fixed x_train, and x_batch = x_train[permutation[i * bs: (i + 1) * bs]]
    matching = [None for _ in range(n)]
    nat_force_active = False

    for epoch in range(args.nb_epoch):
        random_permutation = np.random.permutation(n)

        
        if epoch % 10 == 0 and epoch > 20:
            K.set_value(nat_loss_weight_variable, 10 * (epoch-30))
            nat_force_active = True

        if epoch % 10 != 0 and epoch > 20:
            nat_force_active = False


        for i in range(iters_in_epoch):
            bs = args.batch_size
            indices = random_permutation[i * bs: (i + 1) * bs]
            x_batch = x_train[indices]

            if K.get_value(nat_loss_weight_variable) == 0:
                nat_indices = [matching[latent_index] if matching[latent_index] is not None else np.random.randint(n)
                for latent_index in indices]
            else:
                nat_indices = []
                numnone = 0
                for latent_index in indices:
                    assert numnone < 0.9 * n, "Sok None"
                    if matching[latent_index] is not None:
                        nat_indices.append(matching[latent_index])
                    else:
                        nat_indices.append(np.random.randint(n))
                        numnone += 1
                print("number of none ", numnone)    

            nat_batch = natPositions[nat_indices]

            res = models.ae.train_on_batch([x_batch, nat_batch], x_batch)

            currentLatentPositions = models.encoder.predict(
                [x_batch], batch_size=bs)
            if args.sampling:
                currentLatentPositions = currentLatentPositions[1]  # z_mean!

            latentPositions[indices] = currentLatentPositions

            if not nat_force_active:
                continue

            # if you do both, you can compare. normally you'd only do_approx.
            # if you only want to test that the nat spring force affects the latents,
            # then use matching = list(range(n))

            '''
            if epoch < 60:
                do_exact = True
                do_approx = False
            else:
                do_exact = False
                do_approx = True    
            '''

            do_exact = False
            if do_exact:
                start = time.clock()
                matching_exact = updateBipartiteGraphFromScratch(latentPositions, natPositions)
                print("exact  running time", time.clock() - start)
                print("exact ", len(matching_exact ),
                    weight_of_matching(matching_exact,  latentPositions, natPositions), matching_exact[:10])
                matching = matching_exact
            do_approx = True
            if do_approx:
                start = time.clock()
                matching_approx = updateBipartiteGraph(indices, latentPositions, natPositions, bipartite, annoyIndex)
                print("approx running time", time.clock() - start)
                print("approx", len(matching_approx),
                    weight_of_matching(matching_approx, latentPositions, natPositions), matching_approx[:10])
                matching = matching_approx

            # WARNING don't put functionality here, skipped if not nat_force_active.

        plot_matching(latentPositions, natPositions, matching, "%s/matching-%03d" % (args.outdir, epoch))

        currentEpochPos = models.encoder.predict(images, args.batch_size)
        z_sampled, z_mean, z_logvar = currentEpochPos

        # TODO move ellipse vis to its own function and vis.py
        inum = len(images)

        ells = [Ellipse(xy=z_mean[i],
                        width=2 * np.exp(z_logvar[i][0]),
                        height=2 * np.exp(z_logvar[i][1]))
                for i in range(inum)]

        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

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

        plt.xlim(-5,5)
        plt.ylim(-5,5)

        plt.title("%s/latent-%03d" % (args.outdir, epoch))
        plt.savefig("{}/latent-{:03d}".format(args.outdir, epoch))

        plt.clf()
        plt.close()

        print('epoch: {:03d}/{:03d}, iters: {:03d}/{:03d}'.format(epoch,
                args.nb_epoch, i, iters_in_epoch), end='')
        for (key, value) in zip(models.ae.metrics_names, res):
                print(", ", key, ": ", value, end='')
        print("\n", end='')

        imageDisplayCallback.on_epoch_end(epoch, logs=None)

    # save models
    model_IO.save_autoencoder(models, args)


def build_models(args):
    loss_features = AttrDict({})

    if args.sampling:
        encoder_output_shape = (args.latent_dim, 2)
    else:
        encoder_output_shape = (args.latent_dim, )

    if args.encoder == "dense":
        encoder = dense.build_model(args.original_shape,
                                    encoder_output_shape,
                                    args.encoder_dims,
                                    args.encoder_wd,
                                    args.encoder_use_bn,
                                    args.activation,
                                    "linear")
    elif args.encoder == "conv":
        encoder = conv.build_model(args.original_shape, encoder_output_shape, args.encoder_conv_channels,
                                   args.encoder_wd, args.encoder_use_bn, args.activation, "linear")
    elif args.encoder == "conv_deconv":
        encoder = conv.build_model_conv_encoder(args.original_shape, encoder_output_shape,
                                                args.encoder_conv_channels, args.encoder_wd, args.encoder_use_bn, args.activation, "linear")
    else:
        assert False, "Unrecognized value for encoder: {}".format(args.encoder)

    generator_input_shape = (args.latent_dim, )
    if args.generator == "dense":
        generator = dense.build_model(generator_input_shape,
                                      args.original_shape,
                                      args.generator_dims,
                                      args.generator_wd,
                                      args.generator_use_bn,
                                      args.activation,
                                      "linear")
    elif args.generator == "conv":
        generator = conv.build_model(generator_input_shape, args.original_shape, args.generator_conv_channels,
                                     args.generator_wd, args.generator_use_bn, args.activation, "linear")
    elif args.generator == "conv_deconv":
        generator = conv.build_model_conv_decoder(generator_input_shape, args.original_shape,
                                                  args.generator_conv_channels, args.generator_wd, args.generator_use_bn, args.activation, "linear")
    else:
        assert False, "Unrecognized value for generator: {}".format(
            args.generator)

    if args.sampling:
        sampler_model = add_gaussian_sampling(encoder_output_shape, args)

        inputs = Input(shape=args.original_shape)
        nats = Input(shape=(args.latent_dim, ))
        hidden = encoder(inputs)
        (z, z_mean, z_log_var) = sampler_model(hidden)
        encoder = Model(inputs, [z, z_mean, z_log_var])

        loss_features["z_mean"] = z_mean
        loss_features["z_log_var"] = z_log_var
        loss_features["z_nat"] = nats
        output = generator(z)
        ae = Model([inputs, nats], output)
    else:
        ae = Sequential([encoder, generator])

    modelDict = AttrDict({})
    modelDict.ae = ae
    modelDict.encoder = encoder
    modelDict.generator = generator

    return modelDict, loss_features


def add_gaussian_sampling(input_shape, args):
    assert input_shape[-1] == 2
    inputs = Input(shape=input_shape)

    z_mean = Lambda(lambda x: x[..., 0], output_shape=input_shape[:-1])(inputs)
    z_log_var = Lambda(lambda x: x[..., 1],
                       output_shape=input_shape[:-1])(inputs)

    output_shape = list(K.int_shape(z_mean))
    output_shape[0] = args.batch_size

    def sampling(inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=output_shape, mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    z = Lambda(sampling)([z_mean, z_log_var])
    sampler_model = Model(inputs, [z, z_mean, z_log_var])
    return sampler_model
