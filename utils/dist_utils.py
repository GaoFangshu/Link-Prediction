"""
@reference: Chenglong Chen (https://github.com/ChenglongChen/Kaggle_HomeDepot/)
@brief: utils for distance computation
"""

import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import lzma
    import Levenshtein
except:
    pass

from utils import np_utils
sys.path.append("..")

def _jaccard_coef(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return np_utils._try_divide(float(len(A.intersection(B))), len(A.union(B)))


def _dice_dist(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return np_utils._try_divide(2.*float(len(A.intersection(B))), (len(A) + len(B)))