# MLp
This repository aims at creating a modular and expendable playground for building ML workflows based on tabular data.
MODULAR, EASY, QUICK AND REPRODUCIBLE DATA PREPROCESSING (Based on sklearn), MODEL BUILDING, FINETUNING AND TRAINING/INFERENCE PIPELINES.
Local install:
- activate your environment 
- run 'pip install git+https://github.com/ValentinOzeel/MLp.git@main#egg=MLp'


MLP_preprocessing: 
- With the MLpPreprocessing class, create and apply data your data preprocessing pipelines. 
- Initial preprocessing strategy experiments can easily be carried out with the MLpPlaygroundPreprocessing class (useful tools such as reverting some applied preprocessing steps).
- Create comprehensive data pipeline with building blocks (sklearn transformer or function transformer) simply defined in a list. 
- Encompasses the DynamicFeatureTransformer and FunctionTransformerTransformer classes that use Sklearn's transformer and functiontransformer as custom building blocks for creating data preprocessing pipelines. 
- Use available transformers (LIST in src/preprocessing/MLp_preprocessing_transformers and src/preprocessing/MLp_sampling_transformers) as building blocks or wrap your custom sklearn transformers (or function transformers) with MLp_preprocessing's DynamicFeatureTransformer and FunctionTransformerTransformer classes to enable various abilities such as applying on all or specific columns (or filtered columns: categoricals, numericals, all cols 'except these'), preprocessing before main transformation, drop initial features
- MLp preprocessing enables to seemlessly apply sampling technique such as outlier removal within the same comprehensive data transformation pipeline. 

MLP_model: Define your model with the MLpModel class, apply calibration, evaluate model score performance and feature importance.

MLP_builder: Aim at enabling the user to quicly run model pipelines for learning, testing or potential deployment. Use your data pipelines and defined model to create a framework for data preprocessing, model training (easy pre-split and split data), finetuning (grid or optuna), cross-validation, inferences and experiment tracking with mlflow.

Integrate well with Kedro projects.
Nvidia rapids cuda and Intel CPU acceleration available for preprocessing and model training.


