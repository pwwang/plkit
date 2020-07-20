"""Exceptions for plkit"""

class PlkitException(Exception):
    """Base exception class for plkit"""

class PlkitDataException(PlkitException):
    """Something wrong when preparing data"""

class PlkitDataSizeException(PlkitException):
    """Dimension of data is wrong"""

class PlkitMeasurementException(PlkitException):
    """Measurement is wrong"""
