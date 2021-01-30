#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

import argparse
from pathlib import Path

from src.data import load_data, save_as_csv


def main(train_path, test_path,
         output_dir):
    """Normalize data"""

    # set vars
    norm_method = {"min_max", "z_score"}

    train_df, test_df, output_dir, params = load_data(train_path,
                                                      test_path,
                                                      output_dir,
                                                      load_params=True)

    # optionally normalize data
    if params["normalize"] in norm_method:
        # TODO add function to normalize data
        raise NotImplementedError

    # save train
    save_as_csv(train_df, train_path, output_dir,
                replace_text="_nan_imputed.csv",
                suffix="_processed.csv",
                na_rep="nan")

    # save test
    save_as_csv(test_df, test_path, output_dir,
                replace_text="_nan_imputed.csv",
                suffix="_processed.csv",
                na_rep="nan")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-te", "--test", dest="test_path",
                        required=True, help="Test CSV file")
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        default=Path("./data/processed").resolve(),
                        required=False, help="output directory")
    args = parser.parse_args()

    # convert categorical variables into integer codes
    main(args.train_path, args.test_path,
         args.output_dir)
