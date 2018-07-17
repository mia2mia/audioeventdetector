# -*- coding: utf-8 -*-

import h5py
import numpy as np
import config

cfg = config.Config()


def load_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        video_id_list = hf.get('video_id_list')
        x = np.array(x)
        y = list(y)
        video_id_list = list(video_id_list)
        
    return x, y, video_id_list


def to_float32(x,y):
    return ((np.float32(x) - 128.) / 128.), np.float32(y)


def create_dataset(x, y):
    counts = {inds:0 for inds in range(cfg.NUM_CLASSES)}
    data = []
    labels = []
    for i, label in enumerate(y):
        if np.sum(label[cfg.CLASS_INDS]) == 1:
            label_id = np.where(label[cfg.CLASS_INDS]==1)[0][0]
            if counts[label_id] > 800:
                continue
            # increase the count for that class
            counts[label_id] += 1
            data.append(x[i])
            labels.append(label[cfg.CLASS_INDS])
    print ("Data count per class: ")
    for k,v in counts.items():
        print ("{}: {}".format(k,v))
    return np.dstack(data).transpose(2,0,1), np.dstack(labels).transpose(2,0,1).squeeze()


if __name__=='__main__':
    print ("Loading Data ...")
    (x_bal, y_bal, _) = load_data(cfg.BAL_HDF5_PATH)
    x_bal, y_bal = to_float32(x_bal, y_bal) # shape: (N, 10, 128)
    (x_unbal, y_unbal, _) = load_data(cfg.UNBAL_HDF5_PATH)
    x_unbal, y_unbal = to_float32(x_unbal, y_unbal) # shape: (N, 10, 128)
    (x_eval, y_eval, _) = load_data(cfg.EVAL_HDF5_PATH)
    x_eval, y_eval = to_float32(x_eval, y_eval) # shape: (N, 527)
    print ("Done ... Creating Dataset...")
    x_train, y_train = create_dataset(np.concatenate((x_bal, x_unbal)), np.concatenate((y_bal, y_unbal)))
    print (x_train.shape, y_train.shape)
    x_val, y_val = create_dataset(x_eval, y_eval)
    print (x_val.shape, y_val.shape)

    np.save('dataset/x_train.npy', x_train)
    np.save('dataset/y_train.npy', y_train)
    np.save('dataset/x_val.npy', x_val)
    np.save('dataset/y_val.npy', y_val)
 

