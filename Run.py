import os
import time
import argparse
import numpy as np
import pandas as pd

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from Models import SNLVM, LFADS, TNDM, NFLVM
from Models.runtime import Runtime, ModelType
from Models.models.model_loader import ModelLoader
from Utils import DataIO
from Utils import Plot

parser = argparse.ArgumentParser()
parser.add_argument('--model',   default='NFLVM', type=str)
parser.add_argument('--times',   default=4,       type=int)
parser.add_argument('--heldout', default=False,   type=bool)
args = parser.parse_args()

if args.model == 'SNLVM':
    model_type = ModelType.SNLVM
    model_class = SNLVM
if args.model == 'LFADS':
    model_type = ModelType.LFADS
    model_class = LFADS
if args.model == 'NFLVM':
    model_type = ModelType.NFLVM
    model_class = NFLVM
if args.model == 'TNDM':
    model_type = ModelType.TNDM
    model_class = TNDM


#%% model parameters

loss_weights = [1.0, 1e-1, 1e-1, 100000000.0]
prior_variance = 1e3
epochs = 1000
batch_size = 16
learning_rate = 1e-2
factors = 4

#%% read the data

[[train_spikes,   val_spikes,   test_spikes,
  train_behavior, val_behavior, test_behavior,
  train_label,    val_label,    test_label], _] = DataIO.ReadData()

[[heldin_train_spikes,   heldin_val_spikes,   heldin_test_spikes,
  heldin_train_behavior, heldin_val_behavior, heldin_test_behavior,
  heldin_train_label,    heldin_val_label,    heldin_test_label],
 [heldout_train_spikes,   heldout_val_spikes,   heldout_test_spikes,
  heldout_train_behavior, heldout_val_behavior, heldout_test_behavior,
  heldout_train_label,    heldout_val_label,    heldout_test_label]] = DataIO.ReadData(True)


#%% run

if not args.heldout :

    R2   = []
    PNLL = []
    RMSE = []

    for i in range(args.times):
    
        print('lambda {}'.format(loss_weights[-1]))
    
        modeldir = os.path.join(
            './Results', args.model+'_'+time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
        
        model, history = Runtime.train(
            model_type    = model_type,
            model_settings = dict(
                seed           = 85114582161775,
                timestep       = 0.01,
                factors        = factors,
                prior_variance = prior_variance,
            ),
            train_dataset = (train_spikes, train_behavior),
            val_dataset   = (val_spikes, val_behavior),
            modeldir         = modeldir,
            optimizer        = tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss_weights     = loss_weights,
            batch_size       = batch_size,
            epochs           = epochs,
            adaptive_lr      = dict(factor=0.95, patience=10, min_lr=1e-5),
            verbose = 0,
        )

        model.save(modeldir)
        print('model saved in {}'.format(modeldir))
        pd.DataFrame(history.history).to_csv(os.path.join(modeldir, 'log.csv'), index=False)

        if args.model == 'SNLVM':
            log_f, _, (_, mean, _), z = model(train_spikes, training=False)
        if args.model == 'LFADS':
            log_f, (_, mean, _), z = model(train_spikes, training=False)
        if args.model == 'NFLVM':
            log_f, (_, mean, _), _, z = model(train_spikes, training=False)
        if args.model == 'TNDM':
            log_f, _, (_, mean, _), _, (z, _) = model(train_spikes, training=False)

        # embedding
        Plot.plot_embeddings(mean, train_label, modeldir=modeldir)
        # behaviour reconstruction
        r2 = Plot.plot_behaviour_reconstruction(z, train_behavior, train_label, modeldir=modeldir)
        # factor-time
        Plot.plot_all_1factors(z, train_label, modeldir=modeldir)
        # firing rates
        rmse = Plot.plot_firing_rates(log_f, train_spikes, train_label, modeldir=modeldir)
        # pnll
        pnll = -1 * np.min(history.history['val_loss_PNLL'])

        R2.append(r2)
        PNLL.append(pnll)
        RMSE.append(rmse)

    print('model {}'.format(args.model))
    print('R2 : {}, {}'.format(np.mean(R2), np.std(R2)))
    print('PNLL : {}, {}'.format(np.mean(PNLL), np.std(R2)))
    print('RMSE : {}, {}'.format(np.mean(RMSE), np.std(R2)))


#%% heldout direction

if args.heldout :

    # train
    
    for i in range(args.times):
    
        print('factors {}'.format(factors))
    
        modeldir = os.path.join(
            './Results', args.model+'_'+time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
        
        model, history = Runtime.train(
            model_type    = model_type,
            model_settings = dict(
                seed           = 85114582161775,
                timestep       = 0.01,
                factors        = factors,
                prior_variance = prior_variance,
            ),
            train_dataset = (heldin_train_spikes, heldin_train_behavior),
            val_dataset   = (heldin_val_spikes, heldin_val_behavior),
            modeldir         = modeldir,
            optimizer        = tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss_weights     = loss_weights,
            batch_size       = batch_size,
            epochs           = epochs,
            adaptive_lr      = dict(factor=0.95, patience=10, min_lr=1e-5),
            verbose = 1,
        )
        
        model.save(modeldir)
        
        pd.DataFrame(history.history).to_csv(os.path.join(modeldir, 'log.csv'), index=False)
    
        print('model saved in {}'.format(modeldir))

        # plot
    
        model = ModelLoader.load(modeldir, model_class=model_class)

        if args.model == 'SNLVM':
            log_f, _, (_, mean, _), z = model(train_spikes, training=False)
        if args.model == 'LFADS':
            log_f, (_, mean, _), z = model(train_spikes, training=False)
        if args.model == 'TNDM':
            log_f, _, (_, mean, _), _, (z, _) = model(train_spikes, training=False)
    
        # embedding
        Plot.plot_embeddings(mean, train_label, modeldir=modeldir, heldout=True)
    
        # behaviour reconstruction
        Plot.plot_behaviour_reconstruction(z, train_behavior, train_label, modeldir=modeldir, heldout=True)
    
        # factor-time
        Plot.plot_all_1factors(z, train_label, modeldir=modeldir)
    
        # firing rates
        Plot.plot_firing_rates(log_f, train_spikes, train_label, modeldir=modeldir)
    
        print('factors {}'.format(factors))
