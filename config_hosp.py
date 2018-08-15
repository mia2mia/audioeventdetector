# -*- coding: utf-8 -*-
import os
# Config file to define the predefined parameters
class Config(object):

    BAL_HDF5_PATH = "packed_features/bal_train.h5"
    EVAL_HDF5_PATH = "packed_features/eval.h5"
    UNBAL_HDF5_PATH = "packed_features/unbal_train.h5"
    CSV_LABEL_PATH = "class_labels_indices.csv"

    X_TRAIN_PATH = 'dataset/x_train.npy'
    Y_TRAIN_PATH = 'dataset/y_train.npy'
    X_VAL_PATH = 'dataset/x_val.npy'
    Y_VAL_PATH = 'dataset/y_val.npy'
    X_TEST_PATH = 'dataset/x_test.npy'
    Y_TEST_PATH = 'dataset/y_test.npy'

    TRAINED_MODEL_PATH = 'weights/aed_hosp_model_0718_150750_0.5758.h5'
    # Dictionary containing classes of interest and their mappings
    CLASSES = {
        14: "Screaming",    
        22: "Crying, Sobbing",
        23: "Baby Cry, Infant Cry",
        47: "Cough",
        49: "Sneeze",
        53: "Walk, Footsteps",
        59: "Hiccup",
        60: "Fart",
        400: "Fire Alarm",
        481: "Beep, Bleep",
    }

    LEARNING_RATE = 0.0001
    DROP_LR_FACTOR = 0.5
    REDUCE_LR_ON_PLATEAU = True
    STEP_DECAY = False
    DECAY_EPOCHS = 10.0
    OPTIMIZER = 'adam'
    BATCH_SIZE = 128
    NUM_EPOCHS = 100

    NUM_DENSE_UNITS = 256
    NUM_DENSE_LAYERS = 0

    NUM_LSTM_UNITS = 32
    NUM_LSTM_LAYERS = 1

    DROPOUT_PROB = 0.5
    MAX_SAMPLES_PER_CLASS = 1200

    def __init__(self):
        self.NUM_CLASSES = len(self.CLASSES)
        self.CLASS_INDS = sorted(self.CLASSES.keys())
        if not os.path.exists(os.path.dirname(self.X_TRAIN_PATH)):
            os.makedirs(os.path.dirname(self.X_TRAIN_PATH))

    def display(self):

        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
