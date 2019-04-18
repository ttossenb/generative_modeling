from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Input, Lambda
from keras import backend as K
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import sys

from util import *
import model_IO
import loss
import vis
import samplers
import callbacks

from networks import dense, conv

import NAT_graph

from matplotlib.patches import Ellipse


# TODO refactor move to its own py
def updateBipartiteGraph(dirties, latentPositions, invertHash, bipartite, annoyIndex):
    n_nbrs = 10
    n = len(invertHash)
    totalChange = 0.0
    changed = 0
    for h in dirties:
        i, sample = invertHash[h]
        latent = latentPositions[h]
        closeIndices, closeDistances = annoyIndex.get_nns_by_vector(
            latent, n_nbrs, include_distances=True)
        oldNeighbors = None
        if i in bipartite:
            oldNeighbors = set(bipartite.neighbors(i))
            bipartite.remove_node(i)
        bipartite.add_node(i, bipartite=0)
        newNeighbors = {closeNatIndex + n for closeNatIndex in closeIndices}
        if oldNeighbors is not None:
            inters = len(newNeighbors.intersection(oldNeighbors))
            jaccardDistance = 1 - inters / (2 * len(oldNeighbors) - inters)
            totalChange += jaccardDistance
            changed += 1
        for j in range(len(closeIndices)):
            closeNatIndex = closeIndices[j]
            dist = closeDistances[j]
            bipartite.add_edge(i, closeNatIndex + n, weight=dist)
    # print("Average change", totalChange / (changed+0.01), changed, len(latentPositions))
    # sys.stdout.flush()


# TO BE CONTINUED
def lookupNearestNaTs(indices, annoyIndex):
    closeIndices, closeDistances = annoyIndex.get_nns_by_vector(
        latent, n_nbrs, include_distances=True)
    return natIndices


def hashSample(sample):
    return hash(tuple(sample.flatten()))


def run(args, data):
    # (x_train, x_test) = data

    ((x_train, y_train), (x_test, y_test)) = data
    images = x_train[:1000]
    labels = y_train[:1000]

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
    losses = loss.loss_factory(
        loss_names, args, loss_features, combine_with_weights=True)
    metric_names = sorted(set(args.metrics + tuple(loss_names)))
    metrics = loss.loss_factory(
        metric_names, args, loss_features, combine_with_weights=False)

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
    latentPositions = {}
    movements = []

    invertHash = {}
    for i, sample in enumerate(x_train):
        h = hashSample(sample)
        invertHash[h] = (i, sample)

    n = len(x_train)
    print("creating bipartite graph and annoy index.")
    natPositions, bipartite, annoyIndex = NAT_graph.buildServers(
        n, args.latent_dim)
    print("done.")

    for epoch in range(args.nb_epoch):
        np.random.shuffle(x_train)

        for i in range(iters_in_epoch):
            bs = args.batch_size
            x_batch = x_train[i * bs: (i + 1) * bs]

            # TODO that's a quite roundabout way of doing it:
            indices = []
            for sample in x_batch:
                indx, _ = invertHash[hashSample(sample)]
                indices.append(indx)

            # TO BE CONTINUED
            # natIndices = lookupNearestNaTs(indices, latentPositions, annoyIndex)
            # nat_batch = natPositions[natIndices]
            nat_batch = np.zeros((args.batch_size, args.latent_dim))

            res = models.ae.train_on_batch([x_batch, nat_batch], x_batch)

            currentLatentPositions = models.encoder.predict(
                [x_batch], batch_size=bs)
            if args.sampling:
                currentLatentPositions = currentLatentPositions[1]  # z_mean!
            dirties = set()
            for sample, latent in zip(x_batch, currentLatentPositions):
                h = hashSample(sample)
                dirties.add(h)
                if h in latentPositions:
                    movements.append(np.linalg.norm(
                        latentPositions[h] - latent))
                    if len(movements) > 1000:
                        movements = movements[-1000:]
                latentPositions[h] = latent
            updateBipartiteGraph(dirties, latentPositions,
                                 invertHash, bipartite, annoyIndex)

        currentEpochPos = models.encoder.predict(images, args.batch_size)
        z_sampled, z_mean, z_logvar = currentEpochPos

        n = None

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
        
        plt.xlim(-2,2)
        plt.ylim(-2,2)

        plt.savefig("{}/latent-{}".format(args.outdir, epoch))
        plt.clf()

        print('epoch: {:03d}/{:03d}, iters: {:03d}/{:03d}'.format(epoch,
                args.nb_epoch, i, iters_in_epoch), end='')
        for (key, value) in zip(models.ae.metrics_names, res):
                print(", ", key, ": ", value, end='')
        print("\n", end='')

        if i % args.frequency == 0:
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
