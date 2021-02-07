#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

import argparse

import hyperopt
import numpy as np
from hyperopt import tpe, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.data import load_data, load_params, save_params


def main(train_path, cv_idx_path,
         num_eval=100):
    """"Search for optimal parameters using hyperopt and write
    to params.yml in root dir"""

    # read files
    train_df, cv_idx = load_data([train_path, cv_idx_path],
                                 sep=",", header=0,
                                 index_col="PassengerId")

    # load params
    params = load_params()
    classifier = params["classifier"]
    target_class = params["train_test_split"]["target_class"]

    # get independent variables (features) and
    # dependent variables (labels)
    train_feats = train_df.drop(target_class, axis=1)
    train_labels = train_df[target_class]

    # find optimal parameters for a specific model
    if classifier.lower() == "random_forest":
        best_params = rf_model(train_feats.to_numpy(),
                               train_labels.to_numpy(),
                               random_state=params["random_seed"],
                               num_eval=num_eval)  # cv=split_generator
    else:
        raise NotImplementedError

    # update params
    params["model_params"][classifier] = best_params

    # save
    save_params(params)


def rf_model(x_train, y_train,
             random_state=42,
             num_eval=100):
    """Train a Random Forest model and determine optimal parameters using hyperopt"""

    # load params to create SKF splits using identical params
    # TODO - fix function to accept iter for CV - keep getting index out of range
    params = load_params()
    params_split = params['train_test_split']
    params_split["random_seed"] = params["random_seed"]

    # K-fold split into train and dev sets stratified by train_labels
    # using random seed for reproducibility
    skf = StratifiedKFold(n_splits=params_split['n_split'],
                          random_state=params_split['random_seed'],
                          shuffle=params_split['shuffle'])

    # objective function
    def obj_fnc(params):
        estimator = RandomForestClassifier(**params,
                                           random_state=random_state)
        score = cross_val_score(estimator, x_train, y=y_train,
                                scoring='accuracy',
                                cv=skf.split(x_train, y_train)).mean()

        return {"loss": -score, "status": hyperopt.STATUS_OK}

    # search space
    criterion_list = ["gini", "entropy"]
    max_depth_list = [None, 4, 8, 12, 20]
    min_samples_leaf = range(1, 10)
    min_samples_split = range(2, 10)
    space = {"max_depth": hyperopt.hp.choice("max_depth", max_depth_list),
             "min_samples_leaf": hyperopt.hp.choice("min_samples_leaf", min_samples_leaf),
             "min_samples_split": hyperopt.hp.choice("min_samples_split", min_samples_split),
             "criterion": hyperopt.hp.choice("criterion", criterion_list)
             }

    # create trials instance
    trials = Trials()

    # compute optimal parameters
    # TODO - see if can set random_seed to Trials or fmin
    best_param = hyperopt.fmin(obj_fnc, space,
                               algo=tpe.suggest,
                               max_evals=num_eval,
                               trials=trials,
                               rstate=np.random.RandomState(random_state)
                               )

    # update criterion with text option
    best_param["criterion"] = criterion_list[best_param["criterion"]]
    best_param["max_depth"] = max_depth_list[best_param["max_depth"]]
    best_param["min_samples_leaf"] = min_samples_leaf[best_param["min_samples_leaf"]]
    best_param["min_samples_split"] = min_samples_split[best_param["min_samples_split"]]

    return best_param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-cv", "--cvindex", dest="cv_index",
                        required=True, help="CSV file with train/dev split")
    parser.add_argument("-n", "--num-eval", dest="num_eval",
                        default=100,
                        required=False, help="Number of iterations for hyperopt")
    args = parser.parse_args()

    # train model
    main(args.train_path, args.cv_index,
         args.num_eval)
