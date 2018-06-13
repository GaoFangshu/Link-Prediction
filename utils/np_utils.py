# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for numpy
"""

import sys

import numpy as np
from scipy.stats import pearsonr
from collections import Counter

sys.path.append("..")
# import config

def _try_divide(x, y, val=0.0):
    """try to divide two numbers"""
    if y != 0.0:
        val = float(x) / y
    return val