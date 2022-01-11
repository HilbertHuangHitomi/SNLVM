import os
import numpy as np
from Models.data import DataManager


def ReadData(heldout_direction=False):
    dataset, _ = DataManager.load_dataset(
        directory=os.path.join('./Datasets/Chewie_CO_FF_2016-10-07_pos_M1_spikes'),
        filename='dataset.h5')

    train_spikes = dataset['train_data'].astype('float')
    val_spikes   = dataset['valid_data'].astype('float')
    test_spikes  = dataset['test_data'].astype('float')

    train_behavior = dataset['train_behaviours'].astype('float')
    val_behavior   = dataset['valid_behaviours'].astype('float')
    test_behavior  = dataset['test_behaviours'].astype('float')

    train_label = dataset['train_target_direction']
    val_label   = dataset['valid_target_direction']
    test_label  = dataset['test_target_direction']
    label_all   = np.concatenate((train_label, val_label, test_label))
    train_label = np.array([sorted(set(label_all)).index(i) for i in train_label])
    val_label   = np.array([sorted(set(label_all)).index(i) for i in val_label])
    test_label  = np.array([sorted(set(label_all)).index(i) for i in test_label])

    b_mean = np.mean(np.vstack((train_behavior, val_behavior, test_behavior))[:,0,:],axis=0)
    b_std  = np.std(np.vstack((train_behavior, val_behavior, test_behavior))[:,0,:],axis=0)
    for i in range(2):
        train_behavior[:,:,i] = (train_behavior[:,:,i] - b_mean[i]) / b_std[i]
        val_behavior[:,:,i]   = (val_behavior[:,:,i] - b_mean[i]) / b_std[i]
        test_behavior[:,:,i]  = (test_behavior[:,:,i] - b_mean[i]) / b_std[i]

    if not heldout_direction :
        return [[train_spikes,   val_spikes,   test_spikes,
                train_behavior, val_behavior, test_behavior,
                train_label,    val_label,    test_label],
                None]

    else :
        heldout_train_index = np.where(train_label==0)[0]
        heldout_val_index   = np.where(val_label==0)[0]
        heldout_test_index  = np.where(test_label==0)[0]

        heldout_train_spikes   = train_spikes[heldout_train_index]
        heldout_train_behavior = train_behavior[heldout_train_index]
        heldout_train_label    = train_label[heldout_train_index]
        heldout_val_spikes   = val_spikes[heldout_val_index]
        heldout_val_behavior = val_behavior[heldout_val_index]
        heldout_val_label    = val_label[heldout_val_index]
        heldout_test_spikes   = test_spikes[heldout_test_index]
        heldout_test_behavior = test_behavior[heldout_test_index]
        heldout_test_label    = test_label[heldout_test_index]

        heldin_train_index = np.where(train_label!=0)[0]
        heldin_val_index   = np.where(val_label!=0)[0]
        heldin_test_index  = np.where(test_label!=0)[0]

        heldin_train_spikes   = train_spikes[heldin_train_index]
        heldin_train_behavior = train_behavior[heldin_train_index]
        heldin_train_label    = train_label[heldin_train_index]
        heldin_val_spikes   = val_spikes[heldin_val_index]
        heldin_val_behavior = val_behavior[heldin_val_index]
        heldin_val_label    = val_label[heldin_val_index]
        heldin_test_spikes   = test_spikes[heldin_test_index]
        heldin_test_behavior = test_behavior[heldin_test_index]
        heldin_test_label    = test_label[heldin_test_index]

        return [[heldin_train_spikes,   heldin_val_spikes,   heldin_test_spikes,
                 heldin_train_behavior, heldin_val_behavior, heldin_test_behavior,
                 heldin_train_label,    heldin_val_label,    heldin_test_label],
                [heldout_train_spikes,   heldout_val_spikes,   heldout_test_spikes,
                 heldout_train_behavior, heldout_val_behavior, heldout_test_behavior,
                 heldout_train_label,    heldout_val_label,    heldout_test_label]]
