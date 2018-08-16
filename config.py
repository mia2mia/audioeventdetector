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

    TRAINED_MODEL_PATH = 'weights/aed_model_0714_014821.val-acc-0.6995.h5'
    # Dictionary containing classes of interest and their mappings
    CLASSES = {
        4: "Conversation",
        5: "Narration, Monologue",
        14: "Screaming",
        16: "Laughter",
        23: "Baby Cry, Infant Cry",
        47: "Cough",
        53: "Walk, Footsteps",
        67: "Applause",
        388: "Alarm",
        384: "Typing",
    }

    LEARNING_RATE = 0.0001
    # Learning rate drop factor
    DROP_LR_FACTOR = 0.5
    # Whether to reduce learning rate if validation loss plateaus
    REDUCE_LR_ON_PLATEAU = True
    # Whether to enable step decay of learning rate in which case you need to
    # specify DROP_LR_FACTOR
    STEP_DECAY = False
    # Number of epochs after which to decay the learning rate for step decay
    DECAY_EPOCHS = 10.0
    OPTIMIZER = 'adam'
    BATCH_SIZE = 128
    # Number of epochs to train for
    NUM_EPOCHS = 100

    # Number of dense layers to use after the LSTM layers
    NUM_DENSE_UNITS = 256
    NUM_DENSE_LAYERS = 0

    # Number of Bidirectional LSTM layers to use.
    # Number of units refers to LSTM cells in the forward pass
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
