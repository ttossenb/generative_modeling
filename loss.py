from keras import objectives
from keras import metrics
import keras.backend as K
import tensorflow as tf
import numpy as np


# loss_features is an AttrDict with all sorts of tensors that are different from the input-output
# various models have different mechanisms for populating it
def loss_factory(loss_names, args, loss_features=None, combine_with_weights=True):

    def xent_loss(x, x_decoded):
        loss = args.original_size * objectives.binary_crossentropy(x, x_decoded)
        return K.mean(loss)

    def mse_loss(x, x_decoded):
        loss = args.original_size * objectives.mean_squared_error(x, x_decoded)
        return K.mean(loss)

    def mae_loss(x, x_decoded):
        loss = args.original_size * objectives.mean_absolute_error(x, x_decoded)
        return K.mean(loss)

    def size_loss(x, x_decoded): # pushing the means towards the origo
        loss = 0.5 * K.sum(K.square(loss_features.z_mean), axis=-1)
        return K.mean(loss)

    def variance_loss(x, x_decoded): # pushing the variance towards 1
        loss = 0.5 * K.sum(-1 - loss_features.z_log_var + K.exp(loss_features.z_log_var), axis=-1)
        return K.mean(loss)

    def binary_crossentropy_loss(x, x_decoded):
        loss = objectives.binary_crossentropy(x, x_decoded)
        return K.mean(loss)
    def gan_generator_loss(x, x_decoded):
        loss = - K.log(x_decoded)
        return K.mean(loss)
    def accuracy(x, x_decoded):
        acc = metrics.binary_accuracy(x, x_decoded)
        return K.mean(acc)
    def wasserstein_loss(x, x_decoded):
        return K.mean(x*x_decoded)

    losses = []
    for loss in loss_names:
        losses.append(locals().get(loss))

    if not combine_with_weights:
        return losses
    else:
        weightDict = {}
        for w in args.weights:
            weightDict[w[0]] = w[1]

        def lossFun(x, x_decoded):
            lossValue = 0
            for i in range(len(losses)):
                loss = losses[i]
                lossName = loss_names[i]
                currentLoss = loss(x, x_decoded)
                weight = weightDict.get(lossName, 1.0)
                currentLoss *= weight
                print(lossName, "weight", weight)
                lossValue += currentLoss
            return lossValue
        return lossFun
    
