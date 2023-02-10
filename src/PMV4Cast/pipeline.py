"""
This module contains the project pipeline.
"""


import logging
import os
import sys
import time
from datetime import datetime, timedelta
from threading import Thread

import numpy as np
import pandas as pd
import schedule

from io_data import DBClient
from LSTNet.util.Msglog import LogInit
from ml_model import Model
from simple_average import SimpleAverage

# log = logging.getLogger(logger_name)


class Pipeline(object):
    """Class responsible for pipeline fuctionality"""

    def __init__(self, time_step, horizon, window, model_name, epochs, field_names):
        self.active_control = False
        self.field_names = field_names
        self.time_step = time_step
        self.horizon = horizon
        self.init_model(model_name, horizon, window, epochs)
        self.gc_db_client = DBClient(
            start_time=None,  #'2020-01-30T00:00:00Z',
            host=os.env["DB_HOST"],
            port=os.env["DB_PORT"],
            username=os.env["DB_USERNAME"],
            password=os.env["DB_PASSWORD"],
            database=os.env["DB_DATABASE"],
            window=window,
            horizon=horizon,
            time_step=time_step,
        )

        self.fcst_db_client = DBClient(
            start_time="2020-01-30T00:00:00Z",
            host=os.env["DB2_HOST"],
            port=os.env["DB2_PORT"],
            username=os.env["DB2_USERNAME"],
            password=os.env["DB2_PASSWORD"],
            database=os.env["DB2_DATABASE"],
            window=window,
            horizon=horizon,
            time_step=time_step,
        )
        log.info("Start running the {} model".format(model_name))
        self.run()

    def run(self):
        """Run the main processing pipeline"""
        self.active_control = True
        # self.fcst_db_client.client.query('DROP SERIES FROM "GC parking"')
        # self.fcst_db_client.client.query('DROP SERIES FROM "GC parking" WHERE "tag"=\"{}\"'.format('area'))
        self.train()
        self.predict()
        # schedule.every().hour.at(":01").do(lambda s=self: s.predict())
        # schedule.every(10).minutes.at(":01").do(lambda s=self: s.predict())
        schedule.every().day.at("00:01").do(lambda s=self: s.predict())
        schedule.every().monday.at("00:00").do(lambda s=self: s.train())
        while self.active_control:
            schedule.run_pending()
            time.sleep(1)

    def init_model(self, model_name, horizon, window, epochs):
        """Initialize the model"""
        log.info("Initializing the {} model".format(model_name))
        self.model = dict()
        for field_name in self.field_names:
            if model_name == "lstnet":
                self.model[field_name] = Model(model_name, horizon, window, epochs)
            elif model_name == "simple_average":
                self.model[field_name] = SimpleAverage(model_name, horizon, window)
            else:
                print("Model is not specified! Using simple model.")
                self.model[field_name] = SimpleAverage(model_name, horizon, window)
        return

    def train(self):
        """Retrieves training data from database
        and trains the model
        """
        for field_name in self.field_names:
            log.info(
                "Train {} ... at {}".format(
                    field_name, datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                )
            )
            data = self.gc_db_client.read_db(query_type="train", field_name=field_name)
            self.model[field_name].train_model(data)
        return

    def predict(self):
        """Retrieves past series to make future predictions
        and writes the data to database
        """
        for field_name in self.field_names:
            log.info(
                "Predict {} ... at {}".format(
                    field_name, datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                )
            )
            names = ["_q10", "_q50", "_q90"]
            data = self.gc_db_client.read_db(
                query_type="predict", field_name=field_name
            )
            start = data.index[-1] + timedelta(minutes=int(self.time_step[:-1]))
            end = start + timedelta(
                minutes=(self.horizon - 1) * int(self.time_step[:-1])
            )
            predictions = self.model[field_name].make_predictions(data, start, end)

            for i in range(predictions.shape[0]):
                df = pd.DataFrame(
                    data=predictions[i, :, :],
                    index=pd.date_range(
                        start=start, end=end, freq=self.time_step[:-1] + "T"
                    ),
                    columns=data.columns,
                )
                self.fcst_db_client.write_db(
                    df, "GC parking", field_name + "_forecasted" + names[i], "area"
                )

    def stop(self):
        """Emergency/testing interruption"""
        self.active_control = False
        return


if __name__ == "__main__":
    logger_name = "simple_average"
    log = LogInit(logger_name, os.path.join("..", "..", "logs", logger_name), 20, True)
    log.info("Python version: {}".format(sys.version))

    pp = Pipeline(
        "10m", 144, 144 * 7, logger_name, 1, ["count_occupied", "count_available"]
    )
