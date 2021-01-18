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

def remove_nan(input_filepath, output_dir):
    """Remove rows  with missing values from the dataset"""
    assert (os.path.isfile(input_filepath)), FileNotFoundError
    assert (os.path.isdir(output_dir)), NotADirectoryError

    df = pd.read_csv(input_filepath, sep=",", header=0)

    # drop unused cols
    df = df.drop(columns=["Name", "Cabin"])
    df = df.dropna(axis=0, how="any")

    # save file
    output_filename = os.path.basename(input_filepath.replace(".csv",
                                                              "_clean.csv"))
    output_filepath = os.path.join(output_dir, output_filename)
    df.to_csv(output_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        required=True, help="output directory")
    parser.add_argument("-i", "--input", dest="input_filepath",
                        required=True, help="input CSV file")
    args = parser.parse_args()
    remove_nan(args.input_filepath, args.output_dir)