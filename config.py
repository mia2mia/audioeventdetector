# -*- coding: utf-8 -*-

# Config file to define the predefined parameters
class Config(object):

    BAL_HDF5_PATH = "packed_features/bal_train.h5"
    EVAL_HDF5_PATH = "packed_features/eval.h5"
    UNBAL_HDF5_PATH = "packed_features/unbal_train.h5"
    CSV_LABEL_PATH = "class_labels_indices.csv"

    # Dictionary containing classes of interest and their mappings
    CLASSES = {
        5: "Naration, Monologue",
        14: "Screaming",    
        388: "Alarm",
        67: "Applause",
        23: "Baby Cry, Infant Cry",
        4: "Conversation",
        384: "Typing",
        16: "Laughter",
        47: "Cough",
        53: "Walk, Footsteps",
    }

    LEARNING_RATE = 0.001
    OPTIMIZER = 'adam'
    BATCH_SIZE = 32

    def __init__(self):
        self.NUM_CLASSES = len(self.CLASSES)
        self.CLASS_INDS = sorted(self.CLASSES.keys())

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
