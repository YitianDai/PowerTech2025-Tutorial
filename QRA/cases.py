"""
Created on Tue Jan  9 14:50:36 2024

The code in WindStorm.py provides the methods required to model how windstorms
can impact a power system

@author: Eduardo Alejandro Martinez Cesena
https://www.researchgate.net/profile/Eduardo_Alejandro_Martinez_Cesena
"""
from engines.WindStorm import WindConfig, WindClass


def get_WindClass(conf=None):
    """ Get pyene object."""

    return WindClass(conf)


def get_WindConfig():
    """ Get pyene object."""

    return WindConfig()
