# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np

import keras
from keras.layers import Dense, Dropout, Conv2D
from keras.layers import LSTM, TimeDistributed, Bidirectional
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential, load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

import time
import os
import argparse

import utils
import config
cfg = config.Config()


class AudioEventDetector():
    
    def __init__(self, pre_trained=None):
        if pre_trained is None:
            self.build_model()
            self.define_callbacks()
        else:
            self.model = load_model(pre_trained)
            self.define_callbacks()


    def build_model(self):
        self.model = Sequential()
        # Dense layers for LLD learning

        # Bidirectional LSTM Layers
        self.model.add(Bidirectional(LSTM(cfg.NUM_LSTM_UNITS, return_sequences=True), input_shape=(10, 128)))
        self.model.add(Dropout(cfg.DROPOUT_PROB))

        for _ in range(cfg.NUM_LSTM_LAYERS - 1):
            self.model.add(Bidirectional(LSTM(cfg.NUM_LSTM_UNITS, return_sequences=True)))
            self.model.add(Dropout(cfg.DROPOUT_PROB))
        
        # Average Pooling Layer for combining all time steps' outputs
        self.model.add(GlobalAveragePooling1D())

        # Dense Layers
        for _ in range(cfg.NUM_DENSE_LAYERS):
            self.model.add(Dense(cfg.NUM_DENSE_UNITS, activation='relu'))
            self.model.add(Dropout(cfg.DROPOUT_PROB))
        
        # Final Classification Layer with Softmax activation
        self.model.add(Dense(cfg.NUM_CLASSES, activation='softmax'))

        # Loss Function and Metrics
        if cfg.STEP_DECAY:
            opt = optimizers.Adam(lr=0.0)
        else:
            opt = optimizers.Adam(lr=cfg.LEARNING_RATE)
        self.model.compile(loss='categorical_crossentropy', 
                            optimizer=opt, 
                            metrics=['accuracy'])


    def define_callbacks(self):
        if not os.path.isdir('weights'):
            os.makedirs('weights')
        file_path = 'weights/aed_model_' \
                        + time.strftime("%m%d_%H%M%S") \
                        + '.h5'

        self.callback_list = [
            EarlyStopping(
                monitor='val_acc', patience=15, verbose=1, mode='max'
            ),
            ModelCheckpoint(
                filepath=file_path, monitor='val_acc', save_best_only='True', verbose=1, mode='max'
            )
        ]

        if cfg.REDUCE_LR_ON_PLATEAU:
            self.callback_list.append(
                ReduceLROnPlateau(
                    monitor='val_loss', factor=cfg.DROP_LR_FACTOR, patience=5, min_lr=0.00001, verbose=1
                ),
            )

        if cfg.STEP_DECAY:
            self.callback_list.append(
                LearningRateScheduler(
                    step_decay, verbose=1
                ),
            )


    def fit(self, x_train, y_train,
            batch_size=32, 
            epochs=100,
            validation_data=None, class_weight=None):
        
        history = self.model.fit(
                        x=x_train, y=y_train,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=self.callback_list,
                        validation_data=validation_data
                    )

        utils.plot_metrics(history.history)
        

    def predict(self, x_test):
        return self.model.predict(x_test)


def step_decay(epoch):
    initial_lrate = cfg.LEARNING_RATE
    drop = cfg.DROP_LR_FACTOR
    epochs_drop = cfg.DECAY_EPOCHS
    lrate = initial_lrate * (drop ** (epoch//epochs_drop))
    return lrate


if __name__ == "__main__":
    cfg.display()
    aed = AudioEventDetector()
    aed.model.summary()

    x_train = np.load(cfg.X_TRAIN_PATH)
    y_train = np.load(cfg.Y_TRAIN_PATH)
    x_val = np.load(cfg.X_VAL_PATH)
    y_val = np.load(cfg.Y_VAL_PATH)

    aed.fit(x_train, y_train, 
            batch_size=cfg.BATCH_SIZE, epochs=cfg.NUM_EPOCHS, 
            validation_data=(x_val, y_val))
