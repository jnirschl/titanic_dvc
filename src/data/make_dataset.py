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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from kaggle.api.kaggle_api_extended import KaggleApi
from tableone import TableOne


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


def create_data_dictionary(data_path,
                           report_dir="./reports/figures",
                           output_file="data_dictionary.tex"):
    """Create a data dictionary"""
    assert (os.path.isfile(data_path)), FileNotFoundError
    assert (os.path.isdir(report_dir)), NotADirectoryError
    report_dir = Path(report_dir).resolve()

    # read files - do not specify index column
    df = pd.read_csv(data_path, sep=",", header=0,
                     na_values=["nan"])

    # read column datatypes from params.yaml
    with open("params.yaml", 'r') as file:
        params = yaml.safe_load(file)

    # update params for column data types
    param_dtypes = params["dtypes"]
    param_dtypes["Pclass"] = pd.api.types.CategoricalDtype(categories=[1, 2, 3],
                                                           ordered=True)
    df = df.astype(param_dtypes)

    # Save information about column names, non-null count, and
    # column datatypes
    cols = df.columns.to_list()
    n_cols = np.arange(0, len(cols))
    total_rows = df.shape[0]
    null_count = total_rows - df.isna().sum()
    col_dtype = df.dtypes

    # additional processing for categorical
    category_list = []
    ordered_list = []
    for elem in col_dtype.to_list():
        if isinstance(elem, pd.CategoricalDtype):
            category_list.append(elem.categories.to_list())
            ordered_list.append(str(elem.ordered))
        else:
            category_list.append("")
            ordered_list.append("")

    out_df = pd.DataFrame(data={"#": n_cols, "Column": cols,
                                "Non-null count": null_count.to_numpy(),
                                "Dtype": col_dtype.to_list(),
                                "Categories": category_list,
                                "Ordered": ordered_list})
    # write table to latex
    template = r'''\documentclass[preview]{{standalone}}
    \usepackage{{booktabs}}
    \begin{{document}}
    {}
    \end{{document}}
    '''

    output_file = report_dir.joinpath(output_file)
    with open(output_file, "w") as file:
        file.write(template.format(out_df.to_latex()))

    # convert tex to PDF
    if os.path.isfile(output_file) and sys.platform == "linux":
        subprocess.call(["pdflatex", "--output-directory",
                         report_dir, output_file])

    # create instance of tableone and save summary statistics
    summary_df = df.drop(columns=["PassengerId", "Cabin", "Embarked", "Ticket", "Name"])
    categorical_idx = summary_df.columns[summary_df.dtypes == "category"].to_list()
    sig_digits = {"Age": 1}
    min_max = ["Age", "Fare", "Parch", "SibSp"]
    mytable = TableOne(summary_df,
                       columns=summary_df.columns.to_list(),
                       categorical=categorical_idx,
                       decimals=sig_digits,
                       min_max=min_max)

    # save table one
    # write table to latex
    table_filepath = report_dir.joinpath("table_one.tex")
    with open(table_filepath, "w") as file:
        file.write(template.format(mytable.to_latex()))

    # convert tex to PDF
    if os.path.isfile(table_filepath) and sys.platform == "linux":
        subprocess.call(["pdflatex", "--output-directory",
                         report_dir, table_filepath])

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
    args.output_dir = Path(args.output_dir).resolve()
    train_path = args.output_dir.joinpath(args.train_data)
    test_path = args.output_dir.joinpath(args.test_data)

    # download dataset from kaggle
    download_data(args.competition, args.train_data, args.test_data,
                  output_dir=args.output_dir)

    # create data dictionary
    create_data_dictionary(train_path)
