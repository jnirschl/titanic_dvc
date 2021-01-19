#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

import os
import argparse
import pathlib
import pandas as pd
import numpy as np
import yaml
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# read params from src/models/params.ysml
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)

random_seed = params['random_seed']
test_size = params['test_size']
stratify = params['stratify']
max_depth = params["random_forest"]["max_depth"]
min_samples_leaf = params["random_forest"]["min_samples_leaf"]
min_samples_split = params["random_forest"]["min_samples_split"]


def train(train_path, test_path, output_dir,
          test_size=test_size,
          stratify=None,
          random_seed=random_seed):
    """Train RandomForest model and predict survival on
    Kaggle test set"""
    assert (os.path.isfile(train_path)), FileNotFoundError
    assert (os.path.isfile(test_path)), FileNotFoundError
    assert (os.path.isdir(output_dir)), NotADirectoryError

    # read files
    train_df = pd.read_csv(train_path, sep=",", header=0,
                           index_col="PassengerId")
    test_df = pd.read_csv(test_path, sep=",", header=0,
                           index_col="PassengerId")

    # get independent variables (features) and
    # dependent variables (labels)
    train_feats = train_df.drop("Survived", axis=1)
    train_labels = train_df["Survived"]
    test_feats = test_df

    # set column upon which to stratify, if applicable
    # if stratify is not None:
    #     stratify = train_feats[stratify]

    # split into train and dev using random seed
    x_train, x_dev, y_train, y_dev = train_test_split(train_feats, train_labels,
                                                      test_size=test_size,
                                                      random_state=random_seed)

    # create instance using random seed for reproducibility
    RFmodel = RandomForestClassifier(random_state=random_seed)

    # fit model on training data
    RFmodel.fit(x_train, y_train)

    # check performance on dev set
    yhat_dev = RFmodel.predict(x_dev)
    auc_dev = roc_auc_score(y_dev, yhat_dev)
    acc_dev = accuracy_score(yhat_dev, y_dev)

    # fill nans with column mean on test set
    if any(test_feats.isna().any()):
        test_feats.fillna(value=train_feats.mean()[["Age", "Fare"]],
                          inplace=True)

    # predict on Kaggle test set
    yhat_test = RFmodel.predict(test_feats)
    yhat_test = pd.Series(yhat_test,
                          name="Survived",
                          index=test_feats.index,
                          dtype=np.int)

    # save output
    output_dir = pathlib.Path(output_dir)

    # save submission
    yhat_test.to_csv(output_dir.joinpath("test_submission.csv"),
                     header=True)

    # save metrics
    metrics = json.dumps({"AUC":auc_dev, "Accuracy":acc_dev})
    with open(output_dir.joinpath("metrics.json"), "w") as writer:
        writer.writelines(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        required=True, help="output directory")
    parser.add_argument("-tr", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-te", "--test", dest="test_path",
                        required=True, help="Test CSV file")
    parser.add_argument("-s", "--stratify", dest="stratify",
                        default=None, required=False,
                        help="Stratify when splitting train/dev")
    args = parser.parse_args()

    # train model
    train(args.train_path, args.test_path, args.output_dir,
          random_seed=random_seed, test_size=test_size,
          stratify=stratify)  # , max_depth=max_depth,
    # min_samples_leaf=min_samples_leaf,
    # min_samples_split=min_samples_split)
