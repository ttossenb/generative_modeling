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

import NAT_weighted_graph
import NAT_straight
import NAT_graph # just for NAT_graph.buildServers(), mostly obsoleted.

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
    distances = np.sqrt(NAT_straight.pairwiseSquaredDistances(latentPositions, natPositions))
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

    use_annoy = False
    if not use_annoy:
        distances = np.sqrt(NAT_straight.pairwiseSquaredDistances(latentPositions[dirties], natPositions))

    for dirty_index, dirty in enumerate(dirties):
        latent = latentPositions[dirty]

        if use_annoy:
            closeIndices, closeDistances = annoyIndex.get_nns_by_vector(
                latent, n_nearest, include_distances=True)
        else:
            closeIndices = np.argsort(distances[dirty_index])[:n_nearest]
            closeDistances = []
            for j in closeIndices:
                closeDistances.append(distances[dirty_index][j])

        oldNeighbors = None
        if dirty in bipartite:
            oldNeighbors = set([nei for nei, datadict in bipartite.adj[dirty].items() if datadict['near']])
            bipartite.remove_node(dirty)
        bipartite.add_node(dirty, bipartite=0)
        newNeighbors = {closeNatIndex + n for closeNatIndex in closeIndices}
        if oldNeighbors is not None and len(oldNeighbors) > 0:
            inters = len(newNeighbors.intersection(oldNeighbors))
            jaccardDistance = 1 - inters / (2 * len(oldNeighbors) - inters)
            totalChange += jaccardDistance
            changed += 1
        for j in range(len(closeIndices)):
            closeNatIndex = closeIndices[j]
            dist = closeDistances[j]
            bipartite.add_edge(dirty, closeNatIndex + n, weight= dist * (-1), near=True) # minus because maximum weight
        randomIndices = np.random.choice(n, size=n_random, replace=False)
        for natIndex in sorted(randomIndices):
            dist = np.linalg.norm(natPositions[natIndex] - latent)
            bipartite.add_edge(dirty, natIndex + n, weight= dist * (-1), near=False) # minus because maximum weight
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
    d = args.latent_dim

    print("creating bipartite graph and annoy index.")
    natPositions, bipartite, annoyIndex = NAT_graph.buildServers(
        n, args.latent_dim)
    print("done.")
    # natPositions = np.random.normal(0, 1, (n, d))

    # latentPositions = np.zeros_like(natPositions) ; latentPositions.fill(np.nan)
    latentPositions = np.random.normal(0, 1, (n, d)) # not true, just placeholder

    oo = NAT_straight.OOWrapper(latentPoints=latentPositions, targetPoints=natPositions)

    matching_active = True # no warmup for matching, but warmup for nat_loss
    matching = np.arange(n, dtype=int) # hopefully will be overwritten before nat_loss_weight_variable>0 kicks in

    for epoch in range(args.nb_epoch):
        random_permutation = np.random.permutation(n)

        # TODO actually there are two different kinds of warmups:
        # nat_loss_weight_variable warmup lets the system do its thing without disruption.
        # matching_active warmup is strictly a performance optimization.
        warmup = 10
        # if epoch >= warmup:
        #     matching_active = True
        if epoch >= warmup:
            K.set_value(nat_loss_weight_variable, 2 * (epoch - warmup))
        else:
            K.set_value(nat_loss_weight_variable, 0.0)

        for i in range(iters_in_epoch):
            bs = args.batch_size
            indices = random_permutation[i * bs: (i + 1) * bs]
            x_batch = x_train[indices]

            nat_indices = matching[indices]

            nat_batch = natPositions[nat_indices]

            res = models.ae.train_on_batch([x_batch, nat_batch], x_batch)

            currentLatentPositions = models.encoder.predict(
                [x_batch], batch_size=bs)
            if args.sampling:
                currentLatentPositions = currentLatentPositions[1]  # z_mean!

            latentPositions[indices] = currentLatentPositions

            if not matching_active:
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

            do_smart = True
            if do_smart:
                oo.updateBatch(indices, currentLatentPositions)
                print("epoch", epoch, "nat matching weight", oo.evaluateMatching())
                matching = oo.matching

            do_exact = False
            if do_exact:
                start = time.clock()
                matching_exact = updateBipartiteGraphFromScratch(latentPositions, natPositions)
                print("exact  running time", time.clock() - start)
                print("exact ", len(matching_exact ),
                    weight_of_matching(matching_exact,  latentPositions, natPositions), matching_exact[:10])
                matching = np.array(matching_exact)
                assert matching.dtype == np.int64
            do_approx = False
            if do_approx:
                start = time.clock()
                matching_approx = updateBipartiteGraph(indices, latentPositions, natPositions, bipartite, annoyIndex)
                print("approx running time", time.clock() - start)
                print("approx", len(matching_approx),
                    weight_of_matching(matching_approx, latentPositions, natPositions), matching_approx[:10])
                matching = matching_approx

            # WARNING don't put functionality here, skipped if not matching_active.

        max_nr_of_points = 1000
        matchingTruncated = matching[:max_nr_of_points]
        latentPositionsTruncated = latentPositions[:max_nr_of_points]
        natPositionsTruncated = natPositions[matchingTruncated]
        plot_matching(latentPositionsTruncated, natPositionsTruncated, range(len(latentPositionsTruncated)), "%s/matching-%03d" % (args.outdir, epoch))

        currentEpochPos = models.encoder.predict(images, args.batch_size)
        z_sampled, z_mean, z_logvar = currentEpochPos

        # TODO move ellipse vis to its own function and vis.py
        inum = len(images)

        minimal_radius = 0.05
        ells = [Ellipse(xy=z_mean[i],
                        width =2 * np.clip(np.exp(z_logvar[i][0]), minimal_radius, None),
                        height=2 * np.clip(np.exp(z_logvar[i][1]), minimal_radius, None))
                for i in range(inum)]

        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

       
        colored = [[] for _ in range(10)]

        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'teal', 'brown']

        for i in range(inum):
            ax.add_artist(ells[i])
            ells[i].set_clip_box(ax.bbox)
            ells[i].set_alpha(0.5)
            for j in range(10):
                if labels[i] == j:
                    ells[i].set_facecolor(colors[j])
                    colored[j].append(ells[i])
        
        ax.legend([c[0] for c in colored], range(10), loc="best")

        
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

        close = []
        numclose = 10

        distcent = []
        labelset = np.unique(labels)
        ratios_per_labels = {label: [] for label in labelset}
        ratios = []
        confusion_matrix = np.zeros((10, 10), dtype=int)

        for i in range(inum):
            label = labels[i]
            distances = np.linalg.norm(z_mean - z_mean[i], axis=1)
            nearests = np.argsort(distances)[:numclose]
            nearest_labels = labels[nearests]
            for n_l in nearest_labels:
                confusion_matrix[label, n_l] += 1
            nr_of_same_label = sum(nearest_labels == label)
            ratio = float(nr_of_same_label) / numclose
            ratios.append(ratio)
            ratios_per_labels[label].append(ratio)

        for l in labelset:
            print("clustering of label %d: %f" % (l, np.mean(np.array(ratios_per_labels[l]))))
        print("global clustering: %f" % np.mean(np.array(ratios)))
        print("confusion:")
        print(confusion_matrix)


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
