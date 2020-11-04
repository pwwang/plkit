"""Exceptions for plkit"""

class PlkitException(Exception):
    """Base exception class for plkit"""

class PlkitDataException(PlkitException):
    """Something wrong when preparing data"""

class PlkitConfigException(PlkitException):
    """When certain config items are missing"""
