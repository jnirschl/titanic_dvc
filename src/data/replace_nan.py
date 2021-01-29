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

import yaml

from src.data import load_data


def main(train_path, test_path,
         output_dir):
    """Split data into train, dev, and test"""

    train_df, test_df, output_dir, params = load_data(train_path,
                                                      test_path,
                                                      output_dir,
                                                      load_params=True)

    # fill nans with column mean/mode on test set
    mean_age = float(round(train_df["Age"].mean(), 4))
    train_df["Age"].fillna(value=mean_age,
                           inplace=True)
    test_df["Age"].fillna(value=mean_age,
                          inplace=True)

    # update params and save imputation scheme
    params["imputation"] = {"Age": mean_age}
    new_params = yaml.safe_dump(params)

    with open("params.yaml", "w") as writer:
        writer.write(new_params)

    # set output filenames
    save_train_fname = os.path.basename(train_path.replace("_categorized.csv", "_nan_imputed.csv"))
    save_test_fname = os.path.basename(test_path.replace("_categorized.csv", "_nan_imputed.csv"))

    # save imputed dataframes
    train_df.to_csv(output_dir.joinpath(save_train_fname))
    test_df.to_csv(output_dir.joinpath(save_test_fname))


if __name__ == "__main__":
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
