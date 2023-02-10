"""
This module contains machine learning model class
"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    History,
    ModelCheckpoint,
    TerminateOnNaN,
)

from features import quantize
from LSTNet.lstnet_datautil import DataUtil
from LSTNet.lstnet_model import (
    LSTNetModel,
    ModelCompile,
    PostARTrans,
    PostSkipTrans,
    PreARTrans,
    PreSkipTrans,
)
from LSTNet.lstnet_plot import AutoCorrelationPlot, PlotHistory, PlotPrediction
from LSTNet.lstnet_util import GetArguments, LSTNetInit
from LSTNet.util.model_util import LoadModel, SaveHistory, SaveModel, SaveResults
from LSTNet.util.Msglog import LogInit

tf.random.set_seed(0)
import random

random.seed(0)
np.random.seed(0)


logger_name = "lstnet"
import logging

log = logging.getLogger(logger_name)


custom_objects = {
    "PreSkipTrans": PreSkipTrans,
    "PostSkipTrans": PostSkipTrans,
    "PreARTrans": PreARTrans,
    "PostARTrans": PostARTrans,
}


class Model(object):
    """Class that creates machine learning model"""

    def __init__(self, model_name, horizon, window, epochs):
        self.init = self.init_args()
        self.name = model_name
        self.init.horizon = horizon
        self.init.window = window
        self.init.save = os.path.join("..", "..", "save", model_name)
        self.init.load = os.path.join("..", "..", "save", model_name)
        self.init.epochs = epochs
        self.init.highway = window  # default 24
        # self.init.skip = horizon #default 24
        self.scale = None

    def init_args(self):
        try:
            args = GetArguments()
        except SystemExit as err:
            print("Error reading arguments")
            exit(0)
        init = LSTNetInit(args)
        log = LogInit(logger_name, init.logfilename, init.debuglevel, init.log)
        log.info("Python version: %s", sys.version)
        log.info("Tensorflow version: %s", tf.__version__)
        log.info(
            "Keras version: %s ... Using tensorflow embedded keras",
            tf.keras.__version__,
        )
        init.dump()
        return init

    def validate_model(self, lstnet):
        if lstnet is None:
            log.critical("Model could not be loaded or created ... exiting!!")
            exit(1)
        return

    def train_model(self, rawdata):
        """An abstract training function"""
        Data = self.preprocess_data(rawdata.values)
        self.scale = Data.scale
        self.init.CNNKernel = self.scale.shape[0]
        log.info("Creating model")
        self.model = LSTNetModel(self.init, Data.train[0].shape)
        self.validate_model(self.model)
        lstnet_tensorboard = ModelCompile(self.model, self.init)
        log.info(
            "Training model ... started at {}".format(
                datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        )
        h = train(self.model, Data, self.init, lstnet_tensorboard)
        loss, rse, corr, nrmse, nd = self.model.evaluate(Data.valid[0], Data.valid[1])
        log.info(
            "Validation on the validation set returned: Loss:%f, RSE:%f, Correlation:%f, NRMSE:%f, ND:%f",
            loss,
            rse,
            corr,
            nrmse,
            nd,
        )
        test_result = {"loss": loss, "rse": rse, "corr": corr, "nrmse": nrmse, "nd": nd}
        SaveModel(self.model, self.init.save)
        # SaveResults(self.model, self.init, h.history, test_result, list(test_result.keys()))
        # SaveHistory(self.init.save, h.history)
        log.info(
            "Training is done at {}".format(
                datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        )
        return

    def make_predictions(self, rawdata, start, end):
        """An abstract prediction function"""
        log.info("Load model from %s", self.init.load)
        lstnet = LoadModel(self.init.load, custom_objects)
        self.validate_model(lstnet)
        Data_test = self.normalize_data(rawdata.values)
        log.info(
            "Predict testing data ... started at {}".format(
                datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        )
        Yt_hat = np.array(
            [
                lstnet.predict(Data_test, verbose=0)
                for _ in range(self.init.mc_iterations)
            ]
        )
        q10, q50, q90 = self.postprocess_data([np.mean(Yt_hat, 0), np.std(Yt_hat, 0)])
        log.info(
            "Predict testing data done at {}".format(
                datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            )
        )
        return np.array([q10, q50, q90])

    def preprocess_data(self, rawdata, trainpercent=0.9, validpercent=0.1, normalize=2):
        """A wrapper to create train, validation, and test datasets for model training/testing
        based on influx data
        """
        Data = DataUtil(
            rawdata,
            trainpercent,
            validpercent,
            self.init.horizon,
            self.init.window,
            normalize,
        )
        # If file does not exist, then Data will not have attribute 'data'
        if hasattr(Data, "data") is False:
            print("Could not load data!! Exiting")
            exit(1)
        return Data

    def normalize_data(self, rawdata, predict=True):
        if self.init.normalise == 2:
            for i in range(self.scale.shape[0]):
                if self.scale[i] != 0:
                    rawdata[:, i] = rawdata[:, i] / self.scale[i]
        if predict == True:
            test_set = range(self.init.window, int(rawdata.shape[0]))
            n = len(test_set)
            X = np.zeros((n, self.init.window, rawdata.shape[1]))
            for i in range(n):
                end = test_set[i]
                start = end - self.init.window
                X[i, :, :] = rawdata[start:end, :]
        return X

    def postprocess_data(self, data):
        """A wrapper to rescale the predictions and form quantiles"""
        if self.init.normalise == 2:
            for i in range(self.scale.shape[0]):
                for fl in data:
                    if self.scale[i] != 0:
                        fl[:, i] = fl[:, i] * self.scale[i]

        q10 = quantize(data[0], data[1], 0.45).astype(int).clip(0)
        q50 = quantize(data[0], data[1], 0.5).astype(int).clip(0)
        q90 = quantize(data[0], data[1], 0.55).astype(int).clip(0)
        return q10, q50, q90


def train(model, data, init, tensorboard=None):
    """A wrapper to rescale the predictions and form quantiles"""
    if init.validate == True:
        val_data = (data.valid[0], data.valid[1])
    else:
        val_data = None

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=init.patience,
        verbose=1,
        mode="auto",
    )
    mcp_save = ModelCheckpoint(
        init.save + ".h5",
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
    )
    history = model.fit(
        x=data.train[0],
        y=data.train[1],
        epochs=init.epochs,
        batch_size=init.batchsize,
        validation_data=val_data,
        callbacks=[early_stop, mcp_save, TerminateOnNaN(), tensorboard]
        if tensorboard
        else None,
    )
    return history
