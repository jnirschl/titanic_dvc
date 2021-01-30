#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
from src.data import load_data


def main(train_path, output_dir,
         test_path=None):
    """Split data into train, dev, and test"""
    # read file
    train_df, test_df, output_dir, params = load_data(train_path,
                                                      test_path,
                                                      output_dir,
                                                      load_params=True)

    params_split = params['split_train_dev']
    params_split["random_seed"] = params["random_seed"]

    # get independent variables (features) and
    # dependent variables (labels)
    train_feats = train_df.drop(params_split["target_class"], axis=1)
    train_labels = train_df[params_split["target_class"]]
    test_feats = test_df

    # set column upon which to stratify, if applicable
    if params_split['stratify'] is not None:
        raise NotImplementedError
        # TODO

    # stratified K-fold to split train, dev, and test using random seed
    skf = StratifiedKFold(n_splits=params_split['n_splits'],
                          random_state=params_split['random_seed'],
                          shuffle=True)

    # create splits based on train_labels
    train_idx, test_idx = skf.split(train_feats, train_labels)

    # x_train, x_dev, y_train, y_dev = train_test_split(train_feats, train_labels,
    #                                                   test_size=test_size,
    #                                                   random_state=random_seed)

    if test_path:
        assert (os.path.isfile(test_path)), FileNotFoundError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-te", "--test", dest="test_path",
                        required=True, help="Test CSV file")
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        default=Path("./data/ ").resolve(),
                        required=False, help="output directory")
    parser.add_argument("-r", "--remove-nan", dest="remove_nan",
                        default=False, required=False,
                        help="Remove nan rows from training dataset")
    parser.add_argument("-l", "--label", dest="label_dict_name",
                        default="label_encoding.yaml",
                        required=False, help="Name for dictionary mapping category codes to text")
    args = parser.parse_args()

    # split data into train and dev sets
    main(args.train_path, args.test_path,
         args.output_dir,
         remove_nan=args.remove_nan,
         label_dict_name=args.label_dict_name)
