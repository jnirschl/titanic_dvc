#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
# ======================================================================

import os
from pathlib import Path

import pandas as pd
import yaml


def load_data(train_path, test_path,
              output_dir, load_params=False) -> object:
    """Helper function to load train and test files
     as well as optional param loading

    Args:
        train_path (str):
        test_path (str):
        output_dir (str):
        load_params (bool):

    Returns:
        object: """
    output_dir = Path(output_dir).resolve()

    assert (os.path.isfile(train_path)), FileNotFoundError
    assert (os.path.isfile(test_path)), FileNotFoundError
    assert (os.path.isdir(output_dir)), NotADirectoryError

    # read files
    train_df = pd.read_csv(train_path, sep=",", header=0,
                           index_col="PassengerId")
    test_df = pd.read_csv(test_path, sep=",", header=0,
                          index_col="PassengerId")

    # params defaults to empty dict
    params = {}
    if load_params:
        # read column datatypes from params.yaml
        with open("params.yaml", "r") as file:
            params = yaml.safe_load(file)

    return train_df, test_df, output_dir, params
