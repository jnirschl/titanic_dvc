# Titanic DVC

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](./LICENSE)
------------

## Project Goals

Predict survival on the Kaggle Titanic dataset using DVC for reproducible machine learning

## Introduction

This repository uses [Data Version Control (DVC)](https://dvc.org/) to create a machine learning pipeline and track
experiments. We will use a modified version of
the [Team Data Science Process](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview)
as our Data Science Life cycle template. This repository template is based on
the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science).

In order to start, clone this repository and install [DataVersionControl](https://dvc.org/). Follow the instructions
below to proceed through the data science life cycle using DVC to manage parameters, scripts, artifacts, and metrics.

### 1. Domain understanding/problem definition

Project Charter:

Problem definition: predict survival on the Kaggle Titanic dataset

Dataset location: Kaggle

Preferred tools and languages: SciKit-Learn, TensorFlow, HyperOpt; Python

#### Downloading the dataset

The script [make_dataset.py](src/data/make_dataset.py) will download the dataset from Kaggle, create a data dictionary,
and summarize the dataset using [TableOne](https://pypi.org/project/tableone/). The key artifacts of this stage are
the [raw training and testing datasets](data/raw), the [data_dictionary](reports/figures/data_dictionary.tex), and
the [summary table](/reports/figures/table_one.tex).

In your terminal, use the command-line interface to build the first stage of the pipeline.

``` bash
dvc run -n make_dataset -p dtypes \
-d src/data/make_dataset.py \
-o data/raw/train.csv \
-o data/raw/test.csv \
-o reports/figures/table_one.tex
-o reports/figures/data_dictionary.tex
--desc "Download data from Kaggle, create data dictionary and summary dtable"\
 python3 src/data/make_dataset.py -c titanic -tr train.csv -te test.csv -o "./data/raw"
```

#### Encoding categorical labels as integer classes

The script [encode_labels.py](src/data/encode_labels.py) is an intermediate data processing script that accepts the raw
training data, and the "dtypes" parameter from the params.yaml file. It encodes the columns with categorical variables
as integer values for machine processing and saves the updated dataset and encoding scheme. Importantly, the training
and testing data is processed at the same time to ensure the identical label encoding. Key artifacts from this stage
include the [interim categorized datasets](/data/interim) and
the [label encoding scheme](/data/interim/label_encoding.yaml).

``` bash
dvc run -n encode_labels -p dtypes \
-d src/data/encode_labels.py \
-d data/raw/train.csv \
-d data/raw/test.csv \
-o data/interim/train_categorized.csv \
-o data/interim/test_categorized.csv \
-o data/interim/label_encoding.yaml \
--desc "Convert categorical labels to integer values and save mapping" \
python3 src/data/encode_labels.py -tr data/raw/train.csv -te data/raw/test.csv -o data/interim
```

#### Preparing data

This section involves two scripts to prepare the data for machine learning. First, missing values are imputed from the
training data in [replace_nan.py](/src/data/replace_nan.py) and second the features are normalized
in [normalize_data.py](/src/data/normalize_data.py). Key artifacts from this stage include
the [interim nan-imputed datasets](/data/interim) and the [final processed dataset](/data/processed) after feature
normalization.

##### Replace missing age values using mean imputation

``` bash
dvc run -n impute_nan -p imputation
-d src/data/replace_nan.py
-d data/interim/train_categorized.csv
-d data/interim/test_categorized.csv
-o data/interim/test_nan_imputed.csv
-o data/interim/train_nan_imputed.csv
--desc "Replace missing values for age with mean values from training dataset."
python3 src/data/replace_nan.py -tr data/interim/train_categorized.csv -te data/interim/test_categorized.csv -o data/interim
```

##### Normalize features

``` bash

```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
