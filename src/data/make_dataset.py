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
from kaggle.api.kaggle_api_extended import KaggleApi


def download_data(competition, train_data, test_data,
                  output_dir="./data/raw",
                  credentials=".kaggle/kaggle.json"):
    """Download raw dataset from Kaggle"""
    credentials = Path.home().joinpath(credentials)
    output_dir = Path(output_dir).resolve()

    assert (os.path.isfile(credentials)), FileNotFoundError(credentials)
    assert (os.path.isdir(output_dir)), NotADirectoryError(output_dir)

    api = KaggleApi()
    api.authenticate()

    # downloading from kaggle.com/c/titanic
    api.competition_download_file(competition,
                                  train_data, path=output_dir)
    api.competition_download_file(competition,
                                  test_data, path=output_dir)


def encode_labels(train_path, test_path,
                  output_dir, remove_nan=False,
                  label_dict_name="label_encoding.yaml"):
    """Encode categorical labels as numeric, save the processed
    dataset the label encoding dictionary"""
    output_dir = Path(output_dir).resolve()

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

    # convert to categorical
    df["Embarked"] = df["Embarked"].astype("category")
    df["Sex"] = df["Sex"].astype("category")

    # save dictionary mapping text to categorical label
    embarked_dict = {key: val for key, val in enumerate(df["Embarked"].cat.categories)}
    sex_dict = {key: val for key, val in enumerate(df["Sex"].cat.categories)}

    # transform to categorical codes
    df["Embarked"] = df["Embarked"].cat.codes
    df["Sex"] = df["Sex"].cat.codes

    # return datasets to train and test
    train_df = df.loc[train_df.index, df.columns]
    test_df = df.loc[test_df.index, df.columns[1:]]

    # remove nan (if applicable
    if remove_nan:
        train_df = train_df.dropna(axis=0, how="any")

    # set output filenames
    save_train_fname = os.path.basename(train_path.replace(".csv", "_categorized.csv"))
    save_test_fname = os.path.basename(test_path.replace(".csv", "_categorized.csv"))

    # save updated dataframes
    train_df.to_csv(output_dir.joinpath(save_train_fname))
    test_df.to_csv(output_dir.joinpath(save_test_fname))  # test cols starts from 1 because survival status is hidden

    # save and encoding dictionaries
    encoding_dict = yaml.safe_dump({"Embarked": embarked_dict,
                                    "Sex": sex_dict})
    with open(os.path.join(output_dir, label_dict_name), "w") as writer:
        writer.writelines(encoding_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--competition", dest="competition",
                        required=True, help="Kaggle competition to download")
    parser.add_argument("-tr", "--train_data", dest="train_data",
                        required=True, help="Train CSV file")
    parser.add_argument("-te", "--test_data", dest="test_data",
                        required=True, help="Test CSV file")
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        default=os.path.dirname(Path(__file__).resolve()),
                        required=False, help="output directory")
    args = parser.parse_args()

    # set vars
    output_dir = Path(args.output_dir).resolve()

    # download dataset from kaggle
    download_data(args.competition, args.train_data, args.test_data,
                  output_dir=output_dir)

    # create data dictionary

    # convert categorical variables into integer codes
    train_path = output_dir.joinpath(args.train_data)
    test_path = output_dir.joinpath(args.test_data)
    encode_labels(train_path, test_path,
                  args.output_dir,
                  remove_nan=args.remove_nan,
                  label_dict_name=args.label_dict_name)
