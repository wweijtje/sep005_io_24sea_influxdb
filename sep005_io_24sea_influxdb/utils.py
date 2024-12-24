import datetime
from typing import Union

from pytz import utc
from shorthand_datetime import parse_shorthand_datetime


def handle_timestamp(dt: Union[datetime.datetime, str], dt_id='start'):
    if isinstance(dt, str):
        dt = parse_shorthand_datetime(dt)

    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(f"The '{dt_id}' datetime object must be timezone-aware.")

    if dt.tzinfo != utc:
        dt = dt.astimezone(utc)

    return dt


# def build_flux_query(start: Union[datetime.datetime,str], location:str, stop=None:Union[None, datetime.datetime, str], site_id=None:str, bucket='metrics', duration=600):
