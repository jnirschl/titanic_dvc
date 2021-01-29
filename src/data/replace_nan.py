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


def main(train_path, test_path,
         output_dir):
    """Split data into train, dev, and test"""
    output_dir = Path(output_dir).resolve()

    assert (os.path.isfile(train_path)), FileNotFoundError
    assert (os.path.isdir(output_dir)), NotADirectoryError

    # read file
    train_df = pd.read_csv(train_path, sep=",", header=0,
                           index_col="PassengerId")
    test_df = pd.read_csv(test_path, sep=",", header=0,
                          index_col="PassengerId")

    # read column datatypes from params.yaml
    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)

    # update params for column data types
    param_dtypes = params["dtypes"]
    # param_dtypes["Pclass"] = pd.api.types.CategoricalDtype(categories=[1, 2, 3],
    #                                                        ordered=True)

    # ensure appropriate data types
    # train_df = train_df.astype(param_dtypes)

    # fill nans with column mean/mode on test set
    mean_age = float(round(train_df["Age"].mean(), 4))
    train_df["Age"].fillna(value=mean_age,
                           inplace=True)
    test_df["Age"].fillna(value=mean_age,
                          inplace=True)

    # update params and save imputation scheme
    params["imputation"] = {"Age": mean_age}
    new_params = yaml.safe_dump(params)

    with open("params.yaml", 'w') as writer:
        writer.write(new_params)

    # set output filenames
    save_train_fname = os.path.basename(train_path.replace("_categorized.csv", "_nan_imputed.csv"))
    save_test_fname = os.path.basename(test_path.replace("_categorized.csv", "_nan_imputed.csv"))

    # save imputed dataframes
    train_df.to_csv(output_dir.joinpath(save_train_fname))
    test_df.to_csv(output_dir.joinpath(save_test_fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-te", "--test", dest="test_path",
                        required=True, help="Test CSV file")
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        default=Path("./data/interim").resolve(),
                        required=False, help="output directory")
    args = parser.parse_args()

    # convert categorical variables into integer codes
    main(args.train_path, args.test_path,
         args.output_dir)
