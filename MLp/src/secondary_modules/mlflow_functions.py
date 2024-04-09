import mlflow
import pandas as pd

def get_or_create_mlflow_experiment(experiment_name: str) -> str:
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.
    Parameters: experiment_name (str): Name of the MLflow experiment.
    Returns: str: ID of the existing or newly created MLflow experiment.
    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    
class CustomModelForMlflow(mlflow.pyfunc.PythonModel):
    """
    A custom model for MLflow integration that allows preprocessing input data
    using provided preprocess pipelines before making predictions with the model.

    Attributes:
        model: The underlying machine learning model.
        preprocess_pipelines: Preprocessing pipelines to apply to input data.
        preprocess_method: Method for applying preprocessing pipelines to input data.

    Methods:
        predict(model_input): 
            Make predictions on input data using the provided model and preprocessing pipelines.

    """
    def __init__(self, model, preprocess_pipelines, preprocess_method):
        self.model = model
        self.preprocess_pipelines = preprocess_pipelines
        self.preprocess_method = preprocess_method
        
    def predict(self, model_input):
        # model_input is a pandas DataFrame
        # Preprocess the input data using the created data pipelines
        preprocessed_input = self.preprocess_method(X_to_predict=model_input, pipelines_to_apply=self.preprocess_pipelines)
        # Generate the predictions
        predictions = self.model.predict(preprocessed_input)
        # Return the predictions as a pandas DataFrame
        return pd.DataFrame(predictions)

