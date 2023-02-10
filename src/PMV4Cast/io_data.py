"""
This module should contain project specific io functionality.

Loading and saving of files should be deferred to this class for easy and consistent file handling between different
sources and to have a single location where file references are held.
"""
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from influxdb import DataFrameClient

# from __main__ import logger_name
logger_name = "simple_average"
log = logging.getLogger(logger_name)


class DBClient(object):
    """A wrapper around influx database"""

    def __init__(
        self,
        host,
        port,
        username,
        password,
        database,
        window,
        horizon,
        start_time,
        time_step,
    ):
        self.start_time = start_time
        self.window = window
        self.horizon = horizon
        self.time_step = time_step
        self.client = DataFrameClient(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        )

    def read_db(self, query_type="train", field_name=""):
        """Fetches data from database
        Args: query_type: str
        Returns:  dataframe:pd.DataFrame
        """
        log.info("Fetching training data of {} field".format(field_name))
        if query_type == "train":
            if self.start_time is None:
                # 4 weeks rolling window
                t_delta = timedelta(
                    minutes=(self.window * 4 + self.horizon) * int(self.time_step[:-1])
                )
                st_time = (datetime.utcnow() - t_delta).strftime("%Y-%m-%dT%H:%M:%SZ")
                select_clause = self.make_query(
                    field_name=field_name, start_time=st_time, time_step=self.time_step
                )
            else:
                select_clause = self.make_query(
                    field_name=field_name,
                    start_time=self.start_time,
                    time_step=self.time_step,
                )
        elif query_type == "predict":
            t_delta = timedelta(
                minutes=(self.window + self.horizon) * int(self.time_step[:-1])
            )
            st_time = (datetime.utcnow() - t_delta).strftime("%Y-%m-%dT%H:%M:%SZ")
            select_clause = self.make_query(
                field_name=field_name, start_time=st_time, time_step=self.time_step
            )
        res = self.client.query(select_clause)
        df_list, df_cols = [], []
        for key, column in res.items():
            df_list.append(res[key])
            df_cols.append(key[1][0][1])
        data = pd.concat(df_list, axis=1).fillna(method="bfill").astype(int)
        data.columns = df_cols
        # convert utc time to local time
        # data.index = data.index.tz_convert('Europe/Helsinki').tz_localize(None)
        # drop tz info
        data.index = data.index.tz_localize(None)
        return data

    def write_db(self, df, measurement_name, field_name, tag_name):
        """Writes to database
        Args:
        Returns:
        """
        log.info("Writing {} data to {} field".format(measurement_name, field_name))
        for column_name in list(df.columns.astype(str)):
            self.client.write_points(
                df[[column_name]].rename(columns={column_name: field_name}),
                measurement=measurement_name,
                field_columns=[field_name],
                tags={"global_tag": tag_name, tag_name: column_name},
                protocol="line",
            )
        return

    def make_query(
        self,
        field_name="count_occupied",
        measurement_name="parking",
        start_time="now()",
        end_time="now()",
        time_step="60m",
        tag_name="area",
    ):
        """A wrapper function to generate string clauses for InfluxDBClient"""
        select_clause = (
            'SELECT  mean("{}") FROM "{}" '
            "WHERE time >= '{}' AND time <= {} - {} "
            'GROUP BY time({}), "{}" fill(previous)'
        ).format(
            field_name,
            measurement_name,
            start_time,
            end_time,
            time_step,
            time_step,
            tag_name,
        )
        return select_clause

    def flush_db(self, measurement_name="GC parking", tag="area"):
        """WARNING! Does NOT work properly with tag"""
        log.info(
            "Flushing database for {} measurement".format(measurement_name, field_name)
        )
        self.client.query(
            'DROP SERIES FROM "{}" WHERE "tag" = \'{}\''.format(measurement_name, tag)
        )
