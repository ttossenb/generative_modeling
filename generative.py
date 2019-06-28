import numpy as np
import keras
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')

import params
import data

import autoencoder
import gan

# load parameters
args = params.getArgs()
print("\n\n")
for k in args.keys():
    print(k, ": ", args[k])
print("\n\n")

# limit memory usage
print("Keras version: ", keras.__version__)
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.memory_share
set_session(tf.Session(config=config))

# load data
data_object = data.load(args.dataset, shape=args.shape, color=args.color)
#(x_train, x_test) = data_object.get_data(args.trainSize, args.testSize)
((x_train, y_train), (x_test, y_test)) = data_object.get_data(args.trainSize, args.testSize)
args.original_shape = x_train.shape[1:]
args.original_size = np.prod(args.original_shape)


if args.model_type == "autoencoder":
    #autoencoder.run(args, (x_train, x_test))
    autoencoder.run(args, ((x_train, y_train), (x_test, y_test)))
elif args.model_type in ("gan", "wgan","wgan-gp"):
    gan.run(args, (x_train, x_test))
else:
    assert False, "Unrecognized model_type: {}".format(args.model_type)


