import os
import sys

import numpy as np
import pandas as pd

pd.options.display.float_format = "{:,.2f}".format
import logging
from datetime import datetime, timedelta

from features import quantize

# from __main__ import logger_name
logger_name = "simple_average"
log = logging.getLogger(logger_name)


class SimpleAverage(object):
    """Makes the predictions based on weekday and time history data"""

    def __init__(self, model_name, horizon, window):
        self.model = dict()
        self.name = model_name
        self.horizon = horizon
        self.window = window

    def train_model(self, rawdata):
        """An abstract training function
        retrieves mean and std of all historical values
        """
        mean_all = (
            rawdata.groupby([rawdata.index.weekday, rawdata.index.time]).mean().round(2)
        )
        std_all = (
            rawdata.groupby([rawdata.index.weekday, rawdata.index.time]).std().round(2)
        )
        self.model["train"] = {"mean": mean_all, "std": std_all}
        return

    def make_predictions(self, rawdata, start, end):
        """An abstract prediction function
        Averages mean and std obtained during training and the same day previous week
        """
        mean_prev = rawdata.groupby([rawdata.index.weekday, rawdata.index.time]).mean()
        # std_prev = rawdata.groupby([rawdata.index.weekday, rawdata.index.time]).std()
        self.model["predict"] = {"mean": mean_prev}  # , 'std':std_prev
        df_mean = pd.concat(
            [self.model["train"]["mean"], self.model["predict"]["mean"]]
        )
        df_std = self.model["train"][
            "std"
        ]  # pd.concat([self.model['train']['std'].pow(2), self.model['predict']['std'].pow(2)])
        mean = df_mean.groupby(level=[0, 1]).mean().round(2)
        # std = df_std.groupby(df_std.index).sum().pow(1/2).round(2)
        mean = mean[(start.weekday(), start.time()) : (end.weekday(), end.time())]
        std = self.model["train"]["std"][
            (start.weekday(), start.time()) : (end.weekday(), end.time())
        ]
        q10, q50, q90 = self.postprocess_data([mean.values, std.values])
        return np.array([q10, q50, q90])

    def postprocess_data(self, data):
        q10 = quantize(data[0], data[1], 0.1).astype(int).clip(0)
        q50 = quantize(data[0], data[1], 0.5).astype(int).clip(0)
        q90 = quantize(data[0], data[1], 0.9).astype(int).clip(0)
        return q10, q50, q90
