#   -*- coding: utf-8 -*-
#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

# imports
import os

import hyperopt
import numpy as np
import pandas as pd
import sklearn
import yaml
from hyperopt import tpe, Trials
from sklearn.model_selection import cross_val_score, train_test_split

# read params from params.ysml
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)

random_seed = params['random_seed']
test_size = params['test_size']
max_depth = params["random_forest"]["max_depth"]
min_samples_leaf = params["random_forest"]["min_samples_leaf"]
min_samples_split = params["random_forest"]["min_samples_split"]


def main(train_path, model="randomforest",
         random_seed=random_seed):
    """"Search for optimal parameters using hyperopt and write
    to params.yml in root dir"""
    assert (os.path.isfile(train_path)), FileNotFoundError
    assert (type(model) is str), TypeError
    assert (model.lower() in ["randomforest"]), TypeError

    # read files
    train_df = pd.read_csv(train_path, sep=",", header=0,
                           index_col="PassengerId")

    # get independent variables (features) and
    # dependent variables (labels)
    train_feats = train_df.drop("Survived", axis=1)
    train_labels = train_df["Survived"]

    # split into train and dev using random seed
    x_train, x_dev, y_train, y_dev = train_test_split(train_feats, train_labels,
                                                      test_size=test_size,
                                                      random_state=random_seed)

    # find optimal parameters for a specific model
    if model.lower() == "randomforest":
        best_params = rf_model(x_train, y_train,
                               random_seed=random_seed,
                               num_eval=100,
                               cross_val=10)
    else:
        raise NotImplementedError

    return best_params


def rf_model(x_train, y_train,
             random_seed=random_seed,
             num_eval=100,
             cross_val=10):
    """Train a Random Forest model and determine optimal parameters using hyperopt"""

    # objective function
    def obj_fnc(params):
        clf = sklearn.ensemble.RandomForestClassifier(**params,
                                                      random_state=random_seed)
        score = cross_val_score(clf, x_train, y=y_train,
                                cv=cross_val).mean()
        return {"loss": -score, "status": hyperopt.STATUS_OK}

    # search space
    criterion_list = ["gini", "entropy"]
    max_depth_list = [None, 4, 8, 12, 20]
    min_samples_leaf = (1, 10)
    min_samples_split = (2, 10)
    space = {"max_depth": hyperopt.hp.choice("max_depth", max_depth_list),
             "min_samples_leaf": hyperopt.hp.choice("min_samples_leaf", min_samples_leaf),
             "min_samples_split": hyperopt.hp.choice("min_samples_split", min_samples_split),
             "criterion": hyperopt.hp.choice("criterion", criterion_list)
             }

    # create trials instance
    trials = Trials()

    # compute optimal parameters
    best_param = hyperopt.fmin(obj_fnc, space,
                               algo=tpe.suggest,
                               max_evals=num_eval,
                               trials=trials,
                               rstate=np.random.RandomState(random_seed)
                               )

    # update criterion with text option
    best_param["criterion"] = criterion_list[best_param["criterion"] - 1]
    best_param["max_depth"] = max_depth_list[best_param["max_depth"]]
    best_param["min_samples_leaf"] = best_param["min_samples_leaf"]
    best_param["min_samples_split"] = best_param["min_samples_split"] + 1

    return best_param


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-tr", "--train", dest="train_path",
    #                     required=True, help="Train CSV file")
    # parser.add_argument("-te", "--test", dest="test_path",
    #                     required=True, help="Test CSV file")
    # parser.add_argument("-s", "--stratify", dest="stratify",
    #                     default=None, required=False,
    #                     help="Stratify when splitting train/dev")
    # args = parser.parse_args()

    # train model
    train_path = "data/interim/train_categorized.csv"
    main(train_path, random_seed=random_seed, test_size=test_size)
