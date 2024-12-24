import configparser
import datetime
import time
from typing import Union, Iterable

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient


from sep005_io_24sea_influxdb.utils import handle_timestamp

class Influx24SEAreader():
    def __init__(self, client:Union[None,InfluxDBClient]=None, bucket:str='metrics', qa:bool=True, verbose:bool=False):
        self.client = client
        self.bucket = bucket
        self.channels = []

        self.qa = qa
        self.verbose = verbose

        self._df = None

    @classmethod
    def from_config_file(cls, config_file:str, **kwargs):

        client = InfluxDBClient.from_config_file(config_file=config_file)
        config = configparser.ConfigParser()
        config.read(config_file)

        for kw in ['bucket', 'qa', 'verbose']:
            if kw not in kwargs:
                # kwargs overrule the configuration file
                if kw in config['influx2']:
                    kwargs[kw] = config['influx2'][kw]

        return cls(client=client, **kwargs)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df:pd.DataFrame):
        new_df.drop(columns=['result', 'table'], inplace=True)
        self._df = new_df
        self._df.set_index('_time', inplace=True)
        self._df.index = pd.to_datetime(self._df.index)
        self._update_properties()  # Also update all related properties

        if self.qa:
            self.missing_samples


    def _update_properties(self):
        """
        Compute properties for ease of use whenever a dataframe is updated

        :return:
        """
        self.channels = [Channel(c, self.df[c], verbose=self.verbose) for c in self.df.columns]
        self.start_timestamp = self.df.index[0]
        self.duration = (self.df.index[-1] - self.start_timestamp).total_seconds()
        self.time = (self.df.index - self.start_timestamp).total_seconds()

    @property
    def missing_samples(self):
        for channel in self.channels:
            channel.missing_samples
        if self.verbose:
            print('QA (missing samples) : Imported signals are equidistant spaced on index')

    def get(self, start, location, stop=None, duration:int=600, sensor_list:Union[None,Iterable]=None):
        """

        :param start:
        :param location:
        :param stop:
        :param bucket:
        :param duration:
        :param sensor_list:
        :return:
        """
        t_0 = time.time()

        query = build_flux_query(
            start=start,
            location=location,
            stop=stop,
            bucket=self.bucket,
            duration=duration,
            sensor_list=sensor_list
        )

        with self.client as client:
            query_api = client.query_api()
            self.df = query_api.query_data_frame(query=query)

        if self.verbose:
            print(f'Time elapsed: {time.time() - t_0}')

    def to_sep005(self):
        """_summary_

        Args:

        Returns:
            list: signals
        """
        signals = []
        for chan in self.channels:
            data = self.df[chan].dropna().to_numpy() # Handle different sample frequencies
            fs_signal = len(data) / self.duration

            signal = {
                'name': chan,
                'data': data,
                'start_timestamp': str(self.start_timestamp),
                'fs': fs_signal,
                'unit_str': '' # This information is not obtainable from the influx db (yet)
            }
            signals.append(signal)

        return signals

class Channel(object):
    def __init__(self, name:str, dfs:pd.Series, verbose=False):
        self.name = name
        self.dfs = dfs.dropna()
        self.data =  self.dfs.to_numpy()
        self.verbose = verbose
        self.start_timestamp = self.dfs.index[0]
        self.time = (self.dfs.index - self.start_timestamp).total_seconds()

        if self.verbose:
            print(f'{self.name}: \t (Fs:{self.fs})')

    @property
    def fs(self):
        return 1 / (self.dfs.index[1] - self.dfs.index[0]).total_seconds()

    @property
    def duration(self):
        return len(self.data) / self.fs
    @property
    def missing_samples(self):
        """
        Check if the sampling frequency is maintained properly
        :return:
        """
        # check the index matches the sampling frequency
        differences = np.diff(self.time)  # Calculate the differences between consecutive elements
        is_equidistant = np.allclose(differences, differences[0] * np.ones(
            len(differences)))  # Check if all differences are the same, up to precision
        if not is_equidistant:
            raise ValueError(f'{self.name}: Samples missing from channel')

def build_flux_query(start, location, stop=None, bucket:str='metrics', duration:int=600, sensor_list:Union[None,Iterable]=None):
    """
    Translates the typical 24SEA nomenclature into an associated influxdb query.

    :param location: location of the sensors
    :param bucket: defaults to 'metrics'
    :param duration: If no `stop` is specified, take duration [s] from start
    :return:
    """

    # check timestamps
    start = handle_timestamp(start, dt_id='start')
    if stop:
        stop = handle_timestamp(stop, dt_id='stop')
    else:
        stop = start + datetime.timedelta(seconds=duration)

    # %% Define the bucket
    query = f"""from(bucket: "{bucket}")"""

    # %% Define the timewindow
    query += f"""|> range(start: {start.isoformat().replace("+00:00", "Z")}, stop:{stop.isoformat().replace("+00:00", "Z")})"""

    # %% Define the filters

    # Only take samples where the "sensor" tag is specified
    query += f"""
    |> filter(fn: (r) => exists r["sensor"])  
    """

    # Isolate on location
    query += f"""
    |> filter(fn: (r) => r["location"] == "{location}")
    """
    # When specified, isolate the sensors based on the sensor_list
    if sensor_list:
        filter_conditions = " or ".join([f'r["sensor"] == "{sensor}"' for sensor in sensor_list])
        query += f"""
         |> filter(fn: (r) => {filter_conditions})
        """

    # Optimise the response
    ### The order of these operations matter! keep->pivot->sort
    query += f"""       
          |> keep(columns: ["_time", "sensor", "_value"])     
          |> pivot(rowKey:["_time"], columnKey: ["sensor"], valueColumn: "_value")
          |> sort(columns: ["_time"], desc:false)   
    """

    return query