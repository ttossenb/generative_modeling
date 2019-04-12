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


# TODO refactor move to its own py
def updateBipartiteGraph(dirties, latentPositions, invertHash, bipartite, annoyIndex):
    n_nbrs = 10
    n = len(invertHash)
    totalChange = 0.0
    changed = 0
    for h in dirties:
        i, sample = invertHash[h]
        latent = latentPositions[h]
        closeIndices, closeDistances = annoyIndex.get_nns_by_vector(latent, n_nbrs, include_distances=True)
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
    print("Average change", totalChange / (changed+0.01), changed, len(latentPositions))
    sys.stdout.flush()


def run(args, data):
    (x_train, x_test) = data
    print(x_test.shape)

    sampler = samplers.sampler_factory(args)

    models, loss_features = build_models(args)
    assert set(("ae", "encoder", "generator")) <= set(models.keys()), models.keys()

    print("Encoder architecture:")
    print_model(models.encoder)
    print("Generator architecture:")
    print_model(models.generator)

    # get losses
    loss_names = sorted(set(args.loss_encoder + args.loss_generator))
    losses = loss.loss_factory(loss_names, args, loss_features, combine_with_weights=True)
    metric_names = sorted(set(args.metrics + tuple(loss_names)))
    metrics = loss.loss_factory(metric_names, args, loss_features, combine_with_weights=False)

    # get optimizer
    if args.optimizer == "rmsprop":
        optimizer = RMSprop(lr=args.lr, clipvalue=1.0)
    elif args.optimizer == "adam":
        optimizer = Adam(lr=args.lr, clipvalue=1.0)
    elif args.optimizer == "sgd":
        optimizer = SGD(lr = args.lr, clipvalue=1.0)
    else:
        assert False, "Unknown optimizer %s" % args.optimizer

    # compile autoencoder
    models.ae.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    imageDisplayCallback = callbacks.ImageDisplayCallback(x_train, x_test, args, models, sampler)

    iters_in_epoch = x_train.shape[0] // args.batch_size
    latentPositions = {}
    movements = []

    invertHash = {}
    for i, sample in enumerate(x_train):
        h = hash(tuple(sample.flatten()))
        invertHash[h] = (i, sample)

    n = len(x_train)
    print("creating bipartite graph and annoy index.")
    natPositions, bipartite, annoyIndex = NAT_graph.buildServers(n, args.latent_dim)
    print("done.")

    for epoch in range(args.nb_epoch):
        np.random.shuffle(x_train)

        for i in range(iters_in_epoch):
            bs = args.batch_size
            x_batch = x_train[i * bs: (i + 1) * bs]
            res = models.ae.train_on_batch(x_batch, x_batch)

            currentLatentPositions = models.encoder.predict(x_batch, batch_size=bs)
            if args.sampling:
                currentLatentPositions = currentLatentPositions[1] # z_mean!
            dirties = set()
            for sample, latent in zip(x_batch, currentLatentPositions):
                h = hash(tuple(sample.flatten()))
                dirties.add(h)
                if h in latentPositions:
                    movements.append(np.linalg.norm(latentPositions[h] - latent))
                    if len(movements) > 1000:
                        movements = movements[-1000:]
                latentPositions[h] = latent
            updateBipartiteGraph(dirties, latentPositions, invertHash, bipartite, annoyIndex)

        print('epoch: {:03d}/{:03d}, iters: {:03d}/{:03d}'.format(epoch, args.nb_epoch, i, iters_in_epoch), end='')
        for (key, value) in zip(models.ae.metrics_names, res):
            print(", ", key, ": ",value, end='')
        print("\n", end='')

        if i % args.frequency == 0:
            imageDisplayCallback.on_epoch_end(epoch, logs=None)

    # save models
    model_IO.save_autoencoder(models, args)

    # save the latent movements
    outputFile = open("latent_movements.npy", "wb")
    arrangedLatentMovements = np.transpose(np.array(positionCollector.latentMovements[1:]))
    np.save(outputFile, arrangedLatentMovements)
    outputFile.close()

    # plot the average latent movements
    plt.plot(np.mean(arrangedLatentMovements, axis=0))
    plt.savefig("pictures/mean_latent_movements.png")
    plt.close()

    latentPositions = np.array(positionCollector.latentPositions)
    from sklearn.decomposition import PCA
    polyline = latentPositions[:, 0, :]
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(polyline)
    print("evr", pca.explained_variance_ratio_)
    plt.plot(reduced[:, 0], reduced[:, 1])
    plt.savefig("pictures/polyline.png")
    plt.close()


    # display randomly generated images
    # vis.displayRandom((10, 10), args, models, sampler, "{}/random".format(args.outdir))

    # display one batch of reconstructed images
    # vis.displayReconstructed(x_train[:args.batch_size], args, models, "{}/train".format(args.outdir))
    # vis.displayReconstructed(x_test[:args.batch_size], args, models, "{}/test".format(args.outdir))


    # # display image interpolation
    # vis.displayInterp(x_train, x_test, args.batch_size, args.latent_dim, encoder, encoder_var, args.sampling, generator, 10, "%s-interp" % args.prefix,
    #                   anchor_indices = data_object.anchor_indices, toroidal=args.toroidal)

    # vis.plotMVhist(x_train, encoder, args.batch_size, "{}-mvhist.png".format(args.prefix))
    # vis.plotMVVM(x_train, encoder, encoder_var, args.batch_size, "{}-mvvm.png".format(args.prefix))



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
        encoder = conv.build_model(args.original_shape, encoder_output_shape, args.encoder_conv_channels, args.encoder_wd, args.encoder_use_bn, args.activation, "linear")
    elif args.encoder == "conv_deconv":
        encoder = conv.build_model_conv_encoder(args.original_shape, encoder_output_shape, args.encoder_conv_channels, args.encoder_wd, args.encoder_use_bn, args.activation, "linear")
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
        generator = conv.build_model(generator_input_shape, args.original_shape, args.generator_conv_channels, args.generator_wd, args.generator_use_bn, args.activation, "linear")
    elif args.generator == "conv_deconv":
        generator = conv.build_model_conv_decoder(generator_input_shape, args.original_shape, args.generator_conv_channels, args.generator_wd, args.generator_use_bn, args.activation, "linear")
    else:
        assert False, "Unrecognized value for generator: {}".format(args.generator)

    if args.sampling:
        sampler_model = add_gaussian_sampling(encoder_output_shape, args)

        inputs = Input(shape=args.original_shape)
        hidden = encoder(inputs)
        (z, z_mean, z_log_var) = sampler_model(hidden)
        encoder = Model(inputs, [z, z_mean, z_log_var])
        
        loss_features["z_mean"] = z_mean
        loss_features["z_log_var"] = z_log_var
        output = generator(z)
        ae = Model(inputs, output)
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

    z_mean = Lambda(lambda x: x[...,0], output_shape=input_shape[:-1])(inputs)
    z_log_var = Lambda(lambda x: x[...,1], output_shape=input_shape[:-1])(inputs)
    
    output_shape = list(K.int_shape(z_mean))
    output_shape[0] = args.batch_size
    
    def sampling(inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=output_shape, mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    z = Lambda(sampling)([z_mean, z_log_var])
    sampler_model = Model(inputs, [z, z_mean, z_log_var])
    return sampler_model
