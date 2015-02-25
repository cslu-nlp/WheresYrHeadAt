from enum import Enum

"""
This module contains an enum for parser actions
"""


class Move(Enum):

    """
    Enum for the parser actions
    """

    shift = 0
    lreduce = 1
    rreduce = 2
