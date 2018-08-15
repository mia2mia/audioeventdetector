# -*- coding: utf-8 -*-
import os
# Config file to define the predefined parameters
class Config(object):

    BAL_HDF5_PATH = "packed_features/bal_train.h5"
    EVAL_HDF5_PATH = "packed_features/eval.h5"
    UNBAL_HDF5_PATH = "packed_features/unbal_train.h5"
    CSV_LABEL_PATH = "class_labels_indices.csv"

    X_TRAIN_PATH = 'dataset_general/x_train.npy'
    Y_TRAIN_PATH = 'dataset_general/y_train.npy'
    X_VAL_PATH = 'dataset_general/x_val.npy'
    Y_VAL_PATH = 'dataset_general/y_val.npy'
    X_TEST_PATH = 'dataset_general/x_test.npy'
    Y_TEST_PATH = 'dataset_general/y_test.npy'

    TRAINED_MODEL_PATH = 'weights/aed_gen_model_0726_135348_0.7181.h5'
    # Dictionary containing classes of interest and their mappings
    CLASSES = {
        4: "Conversation",
        14: "Screaming",    
        16: "Laughter",
        23: "Baby Cry, Infant Cry",
        47: "Cough",
        53: "Walk, Footsteps",
        60: "Fart",
        67: "Applause",
        283: "Wind",
        288: "Water",
        289: "Rain",
        352: "Idling",
        384: "Typing",
        397: "Civil Defense Siren",
        400: "Fire Alarm",
        427: "Gunshot, Gunfire",
        470: "Breaking",
        476: "Rub",
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
