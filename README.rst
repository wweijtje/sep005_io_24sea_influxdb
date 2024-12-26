SEP005 24SEA io influxdb
------------------------

Basic package to import data collected from the 24SEA DF2.0 influx-db compliant with
SDyPy format for timeseries as proposed in SEP005.

Installation
-------------
Important prerequisite
======================

As written in the documentation of the python influx-db client, it is recommended to use ``ciso8601`` with client for
parsing dates. ``ciso8601`` is much faster than built-in Python datetime. Since it's written as a C module the best way is build it from sources:

Easiest way to install is through **conda** :
``conda install -c conda-forge/label/cf202003 ciso8601``



Client configuration
====================
The influxdb client is best configured using the a configuration file. For 24SEA application this can be done using the
project's ``{project_name}_postprocessing.ini`` configuration file.

.. code-block:: ini

    [influx2]
    url=http://influx01.24sea.local:8086/
    org=24sea
    token=my-token
    verify_ssl=False
    bucket=metrics

Using the package
------------------
The key Class is  `Influx24SEAreader` and its '.get()' statement.


    .. code-block:: python

        import datetime
        from pytz import utc
        from sep005_io_24sea_influxdb import Influx24SEAreader

        ini_files = # Add path to {project_name}_postprocessing.ini file
        client = Influx24SEAreader().from_config_file(config_file=ini_file)

        dt = datetime.datetime(2024,11,15,14,tzinfo=utc)
        location = 'WTG001'

        with client:
            client.get(
                start=dt,
                location='WTG01', # Location_id as in influxdb database
                sensor_type='ACC', # type as specified in the influxdb
            )


Once collected the data is stored inside the `client` object. E.g. accessible through
`client.df`. However, it can be directly be exported to sep005 using : `client.to_sep005()`


Acknowledgements
----------------
