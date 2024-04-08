# MLp
This repository aims at creating a modular and expendable playground for building ML projects with sklearn based on tabular data.


MLP_preprocessing: 
Encompass the DynamicFeatureTransformer and FunctionTransformerTransformer classes that enable transformer and functiontransformer use as custom building blocks for data preprocessing. With the MLpPreprocessing class, create and apply data your preprocessing pipelines. 
Initial pipeline building experiment can be easily done with the MLpPlaygroundPreprocessing class (useful tools such as reverting some applied preprocessing steps).
Enable to seemlessly apply specific transformer on specific columns and to apply sampling technique such as outlier removal within the same comprehensive data transformation pipeline. Include data transformers covering splitting/expanding data features to outlier removal (User can write its custom transformer and functiontransformer too).

MLP_model: Define your model with the MLpModel class, apply calibration, evaluate model score performance and feature importance.

MLP_builder: Emcompass the MLpBuilder class that enables quick data initialization, pipeline construction, model training, crossvalidation, hyperparameter tuning (gridsearch or optuna), experiment tracking (hardcoded or mlflow integration), model testing, making inference.

Integrate well with Kedro projects.
Nvidia rapids cuda and Intel CPU acceleration available for preprocessing and model training.


