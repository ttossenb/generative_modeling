import sys # for FlushCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler, Callback
from keras import backend as K
from keras.models import Model
import numpy as np

import vis

class ImageDisplayCallback(Callback):
    def __init__(self, 
                 x_train, x_test, args,
                 modelDict, sampler,
                 **kwargs):
        self.x_train = x_train
        self.x_test = x_test
        self.args = args
        self.modelDict = modelDict
        self.randomPoints = sampler(args.batch_size, args.latent_dim)
        super(ImageDisplayCallback, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs):
        epoch += 1
        if epoch % self.args.frequency != 0:
            return

        randomImages = self.modelDict.generator.predict(self.randomPoints, batch_size=self.args.batch_size)
        vis.plotImages(randomImages, 10, self.args.batch_size // 10, "{}/random-{}".format(self.args.outdir, epoch))
        vis.plotImages(randomImages, 10, self.args.batch_size // 10, "{}/random".format(self.args.outdir))

        trainBatch = self.x_train[:self.args.batch_size]
        vis.displayReconstructed(trainBatch, self.args, self.modelDict, "{}/train-{}".format(self.args.outdir, epoch))
        vis.displayReconstructed(trainBatch, self.args, self.modelDict, "{}/train".format(self.args.outdir))

        testBatch = self.x_test[:self.args.batch_size]
        vis.displayReconstructed(testBatch, self.args, self.modelDict, "{}/test-{}".format(self.args.outdir, epoch))
        vis.displayReconstructed(testBatch, self.args, self.modelDict, "{}/test".format(self.args.outdir))

        #vis.displayInterp(trainBatch, self.args, self.modelDict, gridSize=10, name="{}/interp-{}".format(self.args.outdir, epoch))


class FlushCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        sys.stdout.flush()


class LatentPositionCallback(Callback):
    def __init__(self,
                 x_train, x_test, args,
                 modelDict, sampler,
                 **kwargs):
        self.x_train = x_train
        self.x_test = x_test
        self.args = args
        self.modelDict = modelDict
        self.randomPoints = sampler(args.batch_size, args.latent_dim)
        self.latentPositions = []
        self.latentMovements = []
        super(LatentPositionCallback, self).__init__(**kwargs)

    def on_batch_end(self, batch, logs):
        bs = self.args.batch_size
        currentLatentPositions = self.modelDict.encoder.predict(self.x_train[:bs], batch_size=bs)
        if self.args.sampling:
            currentLatentPositions = currentLatentPositions[1] # z_mean!
        self.latentPositions.append(currentLatentPositions)
        if len(self.latentPositions) > 0:
            currentLatentMovements = np.linalg.norm(currentLatentPositions - self.latentPositions[-1], axis=1)
            self.latentMovements.append(currentLatentMovements)


# def get_lr_scheduler(nb_epoch, base_lr, lr_decay_schedule):
#     assert lr_decay_schedule == sorted(lr_decay_schedule), "lr_decay_schedule has to be monotonically increasing!"

#     def get_lr(epoch):
#         ratio = float(epoch+1) / nb_epoch
#         multiplier = 1.0
#         for etap in lr_decay_schedule:
#             if ratio > etap:
#                 multiplier *= 0.1
#             else:
#                 break
# #         print "*** LR multiplier: ", multiplier
#         return base_lr * multiplier
#     return get_lr

# class SaveGeneratedCallback(Callback):
#     def __init__(self, generator, sampler, prefix, batch_size, frequency, latent_dim, dataset=None, sample_size=100000, save_histogram=False, **kwargs):
#         self.generator = generator
#         self.sampler = sampler
#         self.prefix = prefix
#         self.batch_size = batch_size
#         self.frequency = frequency
#         self.latent_dim = latent_dim
#         self.dataset = dataset
#         self.sample_size = sample_size
#         self.save_histogram = save_histogram
#         super(SaveGeneratedCallback, self).__init__(**kwargs)

#     def save(self, iteration):
#         latent_sample = self.sampler(self.sample_size, self.latent_dim)
#         generated = self.generator.predict(latent_sample, batch_size = self.batch_size)
#         file = "{}_generated_{}.npy".format(self.prefix, iteration)
#         print("Saving generated samples to {}".format(file))
#         np.save(file, generated)
#         if self.save_histogram and self.dataset is not None:
#             vis.print_nearest_histograms(self.dataset, file)

#     def on_epoch_end(self, epoch, logs):
#         if (epoch+1) % self.frequency == 0:
#             self.save(epoch+1)



# class WeightSchedulerCallback(Callback):
#     # weight should be a Keras variable
#     def __init__(self, nb_epoch, name, startValue, stopValue, start, stop, weight, **kwargs):
#         self.nb_epoch = nb_epoch
#         self.name = name
#         self.weight = weight
#         self.startValue = startValue
#         self.stopValue = stopValue
#         self.start = start
#         self.stop = stop
#         super(WeightSchedulerCallback, self).__init__(**kwargs)

#     def on_epoch_end(self, epoch, logs):        
#         phase = 1.0 * (epoch+1) / self.nb_epoch        
#         if phase <= self.start:
#             relative_phase = 0
#         elif phase >= self.stop:
#             relative_phase = 1
#         else:
#             relative_phase = (phase - self.start) / (self.stop - self.start)

#         K.set_value(self.weight, (1-relative_phase) * self.startValue + relative_phase * self.stopValue)
#         print("\nPhase {}, {} weight: {}".format(phase, self.name, K.eval(self.weight)))

# class SaveModelsCallback(Callback):
#     def __init__(self, ae, encoder, encoder_var, generator, prefix, frequency, **kwargs):
#         self.ae = ae
#         self.encoder = encoder
#         self.encoder_var = encoder_var
#         self.generator = generator
#         self.prefix = prefix
#         self.frequency = frequency
#         super(SaveModelsCallback, self).__init__(**kwargs)

#     def on_epoch_end(self, epoch, logs):        
#         if (epoch+1) % self.frequency != 0: return        
#         load_models.saveModel(self.ae, self.prefix + "_model")
#         load_models.saveModel(self.encoder, self.prefix + "_encoder")
#         load_models.saveModel(self.encoder_var, self.prefix + "_encoder_var")
#         load_models.saveModel(self.generator, self.prefix + "_generator")


# class CollectActivationCallback(Callback):
#     def __init__(self, nb_epoch, frequency, batch_size, batch_per_epoch, network, trainSet, testSet, layerIndices, prefix, **kwargs):
#         self.frequency = frequency
#         self.batch_size = batch_size
#         self.network = network
#         self.trainSet = trainSet
#         self.testSet = testSet
#         self.prefix = prefix
#         self.layerIndices = layerIndices
#         self.savedTrain = []
#         self.savedTest = []

#         self.iterations = batch_per_epoch * nb_epoch / frequency
#         self.batch_count = 0

#         outputs = []
#         for i in range(len(self.network.layers)):
#             if i in self.layerIndices:
#                 output = self.network.layers[i].output
#                 outputs.append(output)
#         self.activation_model = Model([self.network.layers[0].input], outputs)
#         train_activations = self.activation_model.predict([self.trainSet], batch_size=self.batch_size)
#         test_activations = self.activation_model.predict([self.testSet], batch_size=self.batch_size)

#         for train_activation, test_activation in zip(train_activations, test_activations):
#             self.savedTrain.append(np.zeros([self.iterations] + list(train_activation.shape)))
#             self.savedTest.append(np.zeros([self.iterations] + list(test_activation.shape)))
#         super(CollectActivationCallback, self).__init__(**kwargs)

#     def on_batch_begin(self, batch, logs):
#         if self.batch_count % self.frequency == 0:
#             train_activations = self.activation_model.predict([self.trainSet], batch_size=self.batch_size)
#             test_activations = self.activation_model.predict([self.testSet], batch_size=self.batch_size)
#             for i in range(len(self.layerIndices)):
#                 self.savedTrain[i][self.batch_count // self.frequency] = train_activations[i]
#                 self.savedTest[i][self.batch_count // self.frequency] = test_activations[i]
#         self.batch_count +=1

#     def on_train_end(self, logs):
#         fileName = "{}_{}.npz".format(self.prefix, self.frequency)
#         outDict = {"train":self.trainSet, "test":self.testSet}
#         for i in range(len(self.layerIndices)):
#             outDict["train-{}".format(self.layerIndices[i])] = self.savedTrain[i]
#             outDict["test-{}".format(self.layerIndices[i])] = self.savedTest[i]
#         print("Saving activation history to file {}".format(fileName))
#         np.savez(fileName, **outDict)

# class ClipperCallback(Callback):
#     def __init__(self, layers, clipValue):
#         self.layers = layers
#         self.clipValue = clipValue

#     def on_batch_begin(self, batch, logs):
#         self.clip()

#     def clip(self):
#         if self.clipValue == 0: return
#         for layer in self.layers:
# #            if layer.__class__.__name__ not in ("Convolution2D"): continue
# #            if layer.__class__.__name__ not in ("BatchNormalization"): continue
#             weights = layer.get_weights()
#             for i in range(len(weights)):
#                 weights[i] = np.clip(weights[i], - self.clipValue, self.clipValue)
#             layer.set_weights(weights)

# class DiscTimelineCallback(Callback):
#     def __init__(self, test_points, batch_size):
#         #self.layers = ayers
#         #self.discriminator = self.model
#         self.test_points = test_points
#         self.batch_size = batch_size
#         self.timeline = []
#         #self.filename = filename

#     def on_epoch_end(self, epoch, logs):
#         self.timeline.append(self.model.predict(self.test_points, batch_size=self.batch_size))
    
# #    def on_epoch_end(self, epoch, logs):
# #        self.saveimg(epoch)

