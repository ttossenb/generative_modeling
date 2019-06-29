import os
import argparse

import params_parse


def getArgs():
    parser = argparse.ArgumentParser()

    # locations
    parser.add_argument('ini_file', nargs='*', help="Ini file to use for configuration")
    parser.add_argument('--outdir', dest="outdir", default="trash", help="Directory for saving output visualizations and models.")

    # which model family?
    parser.add_argument('--model_type', dest="model_type", default="autoencoder", help="What kind of model are we building? See main.py for options.")

    # micellaneous
    parser.add_argument('--memory_share', dest="memory_share", type=float, default=0.45, help="fraction of memory that can be allocated to this process")
    parser.add_argument('--frequency', dest="frequency", type=int, default=20, help="log saving frequency")
    parser.add_argument('--verbose', dest="verbose", type=int, default=2, help="Logging verbosity: 0-silent, 1-verbose, 2-perEpoch (default)")

    # training
    parser.add_argument('--optimizer', dest="optimizer", default="adam", help="Optimizer, adam or rmsprop or sgd")
    parser.add_argument('--lr', dest="lr", default="0.001", type=float, help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', dest="batch_size", default=200, type=int, help="Batch size.")
    parser.add_argument('--nb_epoch', dest="nb_epoch", type=int, default=200, help="Number of epochs")
    parser.add_argument('--loss_encoder', dest="loss_encoder", default="mse_loss", help="comma separated list of losses")
    parser.add_argument('--loss_generator', dest="loss_generator", default="mse_loss", help="comma separated list of losses")
    parser.add_argument('--loss_discriminator', dest="loss_discriminator", default="mse_loss", help="comma separated list of losses")
    parser.add_argument('--metrics', dest="metrics", default="mse_loss", help="comma separated list of metrics")
    parser.add_argument('--weights', dest="weights", default='', help="Comma separated list of (loss_name|loss_weight) pairs. ex size_loss|5,variance_loss|3 means that size_loss has has weight 5 and variance_loss has weight 3")
    
    # dataset
    parser.add_argument('--trainSize', dest="trainSize", type=int, default=0, help="Train set size (0 means default size)")
    parser.add_argument('--testSize', dest="testSize", type=int, default=0, help="Test set size (0 means default size)")
    parser.add_argument('--dataset', dest="dataset", default="celeba", help="Dataset to use")
    parser.add_argument('--color', dest="color", default=True, help="True/False: input color")
    parser.add_argument('--shape', dest="shape", default="64,64", help="comma separated list of image shape")

    
    # architecture
    parser.add_argument('--sampling', dest="sampling", default="False", help="True/False: use sampling")
    parser.add_argument('--activation', dest="activation", default="relu", help="activation function")
    parser.add_argument('--latent_dim', dest="latent_dim", type=int, default=3, help="Latent dimension")

    # encoder
    parser.add_argument('--encoder', dest="encoder", default="dense", help="encoder type")
    parser.add_argument('--encoder_wd', dest="encoder_wd", type=float, default=0.0, help="Weight decay param for the encoder")
    parser.add_argument('--encoder_use_bn', dest="encoder_use_bn", default="False", help="True/False: Use batch norm in encoder")
    parser.add_argument('--encoder_dims', dest="encoder_dims", default="1000,1000", help="Widths of encoder layers")
    parser.add_argument('--encoder_conv_channels', dest="encoder_conv_channels", help="Encoder conv filter counts")

    # generator
    parser.add_argument('--generator', dest="generator", default="dense", help="generator type")
    parser.add_argument('--generator_wd', dest="generator_wd", type=float, default=0.0, help="Weight decay param for generator")
    parser.add_argument('--generator_use_bn', dest="generator_use_bn", default="False", help="True/False> Use batch norm in generator")
    parser.add_argument('--generator_dims', dest='generator_dims', default="1000,1000", help='Widths of generator layers') 
    parser.add_argument('--generator_conv_channels', dest="generator_conv_channels", help="Generator conv filter counts")

    # discriminator
    parser.add_argument('--discriminator', dest="discriminator", default="dense", help="discriminator type")
    parser.add_argument('--discriminator_wd', dest="discriminator_wd", type=float, default=0.0, help="Weight decay param for the discriminator")
    parser.add_argument('--discriminator_use_bn', dest="discriminator_use_bn", default="False", help="True/False: Use batch norm in discriminator")
    parser.add_argument('--discriminator_dims', dest="discriminator_dims", default="1000,1000", help="Widths of discriminator layers")

    # gan
    parser.add_argument('--gan_generator_update', dest="gan_generator_update", type=int, default=1, help="number of generator updates in one step")
    parser.add_argument('--gan_discriminator_update', dest="gan_discriminator_update", type=int, default=1, help="number of discriminator updates in one step")
    
    # parser.add_argument('--clipValue', dest="clipValue", type=float, default=0.01, help="Critic clipping range is (-clipValue, clipValue)")
    # parser.add_argument('--gradient_penalty', dest="gradient_penalty", default="no", help="no/grad/grad_orig")
    
    # parser.add_argument('--modelPath', dest="modelPath", default=None, help="Path to saved networks. If none, build networks from scratch.")
    
    # parser.add_argument('--depth', dest="depth", default=3, type=int, help="Depth of conv vae model")
        

    # parser.add_argument('--lr_decay_schedule', dest="lr_decay_schedule", default='0.5,0.8', help="Comma separated list floats from [0,1] indicating where to decimate the learning rate. Ex 0.2,0.5 means we decimate the learning rate at 20% and 50% of the training")


    default_args_param = parser.parse_args([])
    args_param = parser.parse_args()
    args = params_parse.mergeParamsWithInis(default_args_param, args_param)

    # make sure the following params are tuples
    for k in ("encoder_dims", "generator_dims", "discriminator_dims", "loss_encoder", "loss_generator", "loss_discriminator", "metrics", "shape", "discriminator_conv_channels"):
        if k in args.keys():
            args[k] = ensure_tuple(args[k])
        
    
    params_parse.dumpParams(args, args.outdir + "/all_params.ini")
    return args

def ensure_tuple(v):
    if isinstance(v, str):
        return (v, )
    if not isinstance(v, (tuple,)):
        return (v, )
    return v
