#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

import numpy as np
from sklearn.metrics import precision_score, recall_score


def gmpr_score(y_true, y_pred, weights=None):
    """Compute the geometric mean of precision and recall"""
    # TODO - compare with sklearn FM index
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')

    # update weights parameter and check attributes
    weights = [0.5, 0.5] if weights is None else weights
    assert (type(weights) is list), TypeError
    assert (len(weights) == 2), TypeError

    # compute geometric mean (equally weighted by class)
    gmpr = np.product(np.power([precision, recall], weights))

    return gmpr
