#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
# ======================================================================

import os
import argparse
import pandas as pd
import pathlib

def remove_nan(input_filepath, output_dir):
    """Remove rows  with missing values from the dataset"""
    assert (os.path.isfile(input_filepath)), FileNotFoundError
    assert (os.path.isdir(output_dir)), NotADirectoryError

    df = pd.read_csv(input_filepath, sep=",", header=0)

    # drop unused cols
    df = df.drop(columns=["Name", "Cabin"])

    # remove indices with any nan columns
    df = df.dropna(axis=0, how="any")

    # save file
    output_filename = os.path.basename(input_filepath.replace(".csv",
                                                              "_clean.csv"))
    output_filepath = os.path.join(output_dir, output_filename)
    df.to_csv(output_filepath)

def categorize(train_path, test_path, output_dir):
    """Impute missing values"""
    assert (os.path.isfile(train_path)), FileNotFoundError
    assert (os.path.isfile(test_path)), FileNotFoundError
    assert (os.path.isdir(output_dir)), NotADirectoryError

    # read files
    train_df = pd.read_csv(train_path, sep=",", header=0,
                           index_col="PassengerId")
    test_df = pd.read_csv(test_path, sep=",", header=0,
                          index_col="PassengerId")

    # concatenate df
    df = pd.concat([train_df, test_df], sort=False)

    # drop unnecessary filenames
    df = df.drop(columns=["Name", "Cabin", "Ticket"])

    # set categorical features
    df["EmbarkedId"] = df["Embarked"].astype("category").cat.codes
    df["Sex"] = df["Sex"].astype("category").cat.codes

    # set output columns
    cols = [*df.columns, "EmbarkedId"]

    # save file
    output_dir = pathlib.Path(output_dir)
    save_train_fname = os.path.basename(train_path.replace(".csv", "_categorized.csv"))
    save_test_fname = os.path.basename(test_path.replace(".csv", "_categorized.csv"))
    df.loc[train_df.index, cols].to_csv(output_dir.joinpath(save_train_fname))
    df.loc[test_df.index, cols[1:]].to_csv(output_dir.joinpath(save_test_fname)) # test cols starts from 1 because survival status is hidden


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        required=True, help="output directory")
    parser.add_argument("-r", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-e", "--test", dest="test_path",
                        required=True, help="Test CSV file")
    args = parser.parse_args()
    #remove_nan(args.input_filepath, args.output_dir)

    # convert categorical variables into integer codes
    categorize(args.train_path, args.test_path,
                   args.output_dir)