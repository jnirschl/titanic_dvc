# Titanic DVC

![GitHub](https://img.shields.io/github/license/jnirschl/titanic_dvc)
------------

## Project Goals

Predict survival on the Kaggle Titanic dataset using DVC for reproducible machine learning

## Getting started

This repository uses [Data Version Control (DVC)](https://dvc.org/) to create a machine learning pipeline and track
experiments. We will use a modified version of
the [Team Data Science Process](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview)
as our Data Science Life cycle template. This repository template is based on
the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science).

In order to start, clone this repository and install [DataVersionControl](https://dvc.org/). Next, pip install
requirements according to the script below. Finally, then pull the latest version of data and trained models, which are
hosted on [DagsHub](https://dagshub.com/jnirschl/titanic_dvc).

```bash
# clone the repository
git clone https://github.com/jnirschl/titanic_dvc.git

# create virtual envinroment in folder
cd titanic_dvc
python3 -m venv venv
source venv/bin/activate

# install requirements
pip3 install -r requirements.txt
pip3 install .

# pull data from origin (https://dagshub.com/jnirschl/titanic_dvc)
dvc pull -r origin

# check the status of the pipleline
dvc status

# Expected output
#   Data and pipelines are up to date.
```

## Data science life cycle

### 1. Domain understanding/problem definition

The first step any data science life cycle is to define the question and to understand the problem domain and prior
knowledge. Given a well-formulated question, the team can specify the goal of the machine learning application (e.g.,
regression, classification, clustering, outlier detection) and how it will be measured, and which data sources will be
needed. The scope of the project, key personnel, key milestones, and general project architecture/overview is specified
in the Project Charter and iterated throughout the life of the project. A list of data sources which are available or
need to be collected is specified in the table of data sources. Finally, the existing data is summarized in a data
dictionary that describes the features, number of elements, non-null data, data type (e.g., nominal, ordinal,
continuous), data range, as well as a table with key descriptive summary statistics.

*Deliverables Step 1:*
1. Project charter
2. Table of data sources
3. Data dictionary
4. Summary table of raw dataset

#### Downloading the dataset

The script [make_dataset.py](src/data/make_dataset.py) will download the dataset from Kaggle, create a data dictionary,
and summarize the dataset using [TableOne](https://pypi.org/project/tableone/). The key artifacts of this DVC stage are
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

### 2. Data acquisition and understanding

The second step involves acquiring and exploring the data to determine the quality of the data and prepare the data for
machine learning models. This step involves exploring and cleaning the data to account for missing data and noise as
well as validating that data meet specified validation rules to ensure there were no errors in data collection or data
entry (e.g., age and fare cannot be negative). Once the data is cleaned, it is processed to encode categorical string
variables as integer classes, continuous features are discretized (optional), and features are normalized (optional).
Later stages may iteratively add or create new features from new data or existing features using feature engineering.

*Deliverables Step 2:*
1. Data quality report
2. Proposed data pipeline/architecture
3. Checkpoint decision

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

#### Cleaning data

This section involves preparing the data for machine learning. First, missing values are imputed from the training data
in [replace_nan.py](/src/data/replace_nan.py). Next, feature engineering is used to create additional informative
representations. Once initial feature engineering is complete, the feature set is explored to identify correlated
features and optionally the feature set is reduced using dimensionality reduction techniques. Once a set of feature is
identified, the data is optionally [normalize_data.py](/src/features/normalize.py). Key artifacts from this stage
include the [interim nan-imputed datasets](/data/interim), a [Jupyter notebook](/notebooks)
exploring the dataset and features, [interim feature engineering datasets](/data/interim), and the
[final processed dataset](/data/processed) after feature normalization.

##### Replace missing age values using imputation

``` bash
dvc run -n impute_nan -p imputation
    -d src/data/replace_nan.py
    -d data/interim/train_categorized.csv
    -d data/interim/test_categorized.csv
    -o data/interim/test_nan_imputed.csv
    -o data/interim/train_nan_imputed.csv
    --desc "Replace missing values for age with imputed values from training dataset."
    python3 src/data/replace_nan.py -tr data/interim/train_categorized.csv -te data/interim/test_categorized.csv -o data/interim
```

#### Feature engineering

1. Engineer new features
2. Show feature correlation
3. Identify importance

``` bash
dvc run -n build_features -p random_seed,feature_eng \
    -d src/features/build_features.py
    -d data/interim/train_nan_imputed.csv
    -d data/interim/test_nan_imputed.csv
    -o data/interim/train_featurized.csv
    -o data/interim/test_featurized.csv
    --desc "Optional feature engineering and dimensionality reduction"
    python3 src/features/build_features.py -tr data/interim/train_nan_imputed.csv -te data/interim/test_nan_imputed.csv -o data/interim/  
```

##### Normalize features

``` bash
dvc run -n normalize_data -p normalize \
    -d src/features/normalize.py \
    -d data/interim/train_featurized.csv \
    -d data/interim/test_featurized.csv \
    -o data/processed/train_processed.csv \
    -o data/processed/test_processed.csv \
    --desc "Optionally normalize features by fitting transforms on the training dataset." \
    python3 src/features/normalize.py -tr data/interim/train_featurized.csv -te data/interim/test_featurized.csv -o data/processed/
```

### 3. Modeling

##### Split data into the train, dev, and test sets

+ Data split report

```bash
dvc run -n split_train_dev -p random_seed,train_test_split \
    -d src/data/split_train_dev.py \
    -d data/processed/train_processed.csv \
    -o data/processed/split_train_dev.csv \
    --desc "Split training data into the train and dev sets using stratified K-fold cross validation." \
    python3 src/data/split_train_dev.py -tr data/processed/train_processed.csv  -o data/processed/
```

#### Model training

``` bash
dvc run -n train_model -p classifier,model_params,random_seed,train_test_split.target_class \
    -d src/models/train_model.py \
    -d data/processed/train_processed.csv \
    -d data/processed/split_train_dev.csv \
    -o models/estimator.pkl \
    -m results/metrics.json \
    --desc "Train the specified classifier using the pre-allocated stratified K-fold cross validation splits and the current params.yaml settings." \
    python3 src/models/train_model.py -tr data/processed/train_processed.csv -cv data/processed/split_train_dev.csv
```

#### Predict output

``` bash
dvc run -n predict_output -p predict,train_test_split.target_class \
    -d src/models/predict.py \
    -d src/models/metrics.py \
    -d models/estimator.pkl \
    -d data/processed/test_processed.csv \
    -o results/test_predict_proba.csv \
    -o results/test_predict_binary.csv \
    --desc "Predict output on held-out test set for submission to Kaggle." \
    python3 src/models/predict.py -te data/processed/test_processed.csv -rd results/ -md models/
```

### 4. Deployment

#### Status dashboard

+ Display system health
+ Final modeling report
+ Final solution architecture

### 5. Project conclusion

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
