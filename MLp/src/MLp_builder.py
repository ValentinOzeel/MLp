import os
import numpy as np
from joblib import Parallel, delayed, dump
from typing import Dict, Tuple, List
from colorama import Fore, Style, init
# Initialize colorama to work on Windows as well
init(autoreset=True)

from MLp.src.MLp_preprocessing import MLpPreprocessing
from MLp.src.MLp_model import MLpModel
import MLp.src.secondary_modules.validate_args_kwargs as valid_mlp
from MLp.conf.config_functions import get_config
mlp_config = get_config()
RANDOM_STATE = mlp_config['MLP_CONFIG']['RANDOM_STATE']
from MLp.src.secondary_modules.import_libraries import import_cpu_gpu_sklearn, import_cpu_gpu_pandas

pd = import_cpu_gpu_pandas()
train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV = import_cpu_gpu_sklearn('model_selection', ['train_test_split', 
                                                                                                                     'cross_val_score', 
                                                                                                                     'KFold', 
                                                                                                                     'StratifiedKFold', 
                                                                                                                     'GridSearchCV'])
Pipeline = import_cpu_gpu_sklearn('pipeline', 'Pipeline')
ParameterGrid = import_cpu_gpu_sklearn('model_selection', 'ParameterGrid')
BaseEstimator = import_cpu_gpu_sklearn('base', 'BaseEstimator')

import optuna
# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

import mlflow
mlflow.set_tracking_uri(mlp_config['MLFLOW_CONFIG']['TRACKING_URI'])
from MLp.src.secondary_modules.mlflow_functions import get_or_create_mlflow_experiment
from mlflow.models import infer_signature



# Class to create and run flexible data preprocessing and model training pipelines
class MLpBuilder(MLpPreprocessing, MLpModel):
    def __init__(self, *args, **kwargs):
        self.is_y_continoous = False
        # Define random state
        self.random_state = RANDOM_STATE
        # Best pipeline parameters identified with GridSearch
        self.best_hyperparameters = {}
        
        # Inherit attributes and methods of subclasses
        MLpPreprocessing.__init__(self, *args, **kwargs) 
        MLpModel.__init__(self, *args, **kwargs) 
    
    
    @valid_mlp.validate_initialize_data_intputs
    def initialize_data(self, data=None, target=None, X=None, y=None, 
                        data_copy=False, print_data_info=True, try_y_as_num=True,
                        pre_split=None, split_shuffle=True, split_random_state=None
                        ) -> Tuple:
        
        split_random_state = self.random_state if split_random_state is None else split_random_state
        def _try_y_numerical(y):
            if not try_y_as_num:
                return y   
            else:            
                try: # Check if target is continoous or not. If yes convert it as numerical column
                    y = pd.to_numeric(y)
                    self.is_y_continoous = True
                except ValueError:
                    pass
            return y
        
        def _get_potential_copy(datasets):
            return (my_object.copy() for my_object in datasets) if data_copy else datasets     
            
        if X is None and y is None:  
            self.raw_data = _get_potential_copy((data)) 
            self.y = self.raw_data[target]
            self.X = self.raw_data.drop(target, axis=1)
        else:
            print(type(X), type(y))
            self.X, self.y = _get_potential_copy((X, y)) 
            
        self.y = _try_y_numerical(self.y)  
        
        if print_data_info: self.print_data_info(self.X, self.y)
        
        # Save some data that the model will never see neither for training nor testing. 
        if pre_split is not None and pre_split:
            return self._split_data(self.X, self.y, pre_split, shuffle=split_shuffle, random_state=split_random_state, pre_split=True)

        return self.X, self.y
            

    def print_data_info(self, X, y):
        def summary_df(X_data=None, y_data=None):
            def compute_df(df):
                summary = pd.DataFrame(index=df.columns)
                summary["n_unique"] = df.nunique()
                summary["n_NaN"] = df.isnull().sum()
                summary["Dtype"] = df.dtypes
                return summary
            
            if X_data is not None:
                summary = compute_df(X_data)
            elif y_data is not None:
                if isinstance(y_data, pd.DataFrame):
                    summary = compute_df(y_data)
                elif isinstance(y_data, pd.Series):
                    summary = pd.DataFrame({'n_unique': y_data.nunique(),
                                            'n_NaN': y_data.isnull().sum(),
                                            'Dtype': y_data.dtypes}, 
                                           columns=['n_unique', 'n_NaN', 'Dtype'], index=[y_data.name])
            return summary
            

        title_color, separator_color = Fore.GREEN, Fore.MAGENTA
        def print_section_title(title, title_color=title_color, line_color=separator_color):
            print(f"\n{line_color}{'~' * 40}",
                  f"{title_color}{title:^40}",
                  f"{line_color}{'~' * 40}")
    
        if X is None or y is None:
            X, y = self.X, self.y
            
        print_section_title('X HEAD and TAIL')
        print(f"{X.head()}\n...\n{X.tail()}")
        
        print_section_title('X SHAPE')
        print(f"Rows: {X.shape[0]}, Columns: {X.shape[1]}")

        print_section_title('y SHAPE')
        try:
            y_shape_1 = y.shape[1]
        except (AttributeError, IndexError):
            y_shape_1 = 1
        print(f"Rows: {y.shape[0]}, Column(s): {y_shape_1}")

        print_section_title(f"{'y TYPE'}")
        print(f"{y.dtypes}")

        print_section_title('X DESCRIPTION')
        print(X.describe())

        print_section_title('SUMMARY OF X FEATURES')
        print(summary_df(X_data=X))

        print_section_title('DUPLICATED ROWS IN X')
        print(f"{X.duplicated().sum()}")

        print_section_title('SUMMARY OF Y TARGET')
        print(f"{summary_df(y_data=y)}\n\n{'y VALUE COUNTS':^40}\n{y.value_counts()}")

        print(f"{separator_color}{'~' * 40}{Style.RESET_ALL}")
        return

    
    @valid_mlp.validate__split_data
    def _split_data(self, X, y, test_size, shuffle=True, random_state=None, pre_split=False) -> Tuple:
        """
        Split the data into training and testing/validation sets using train_test_split.
        Parameters:
        - X: DataFrame containing the features.
        - y: DataFrame containing the target.
        - test_size: Size of the test/validation set (e.g., test_size=0.2 -> 80% training, 20% testing).
        - shuffle: Whether to shuffle the data before splitting (default is True).
        Returns:
        - X_train, X_test, y_train, y_test: The split data.
        """
        random_state = self.random_state if random_state is None else random_state
         
        # Separate data into training and testing/validation sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
            )     
        
        if pre_split: 
            if not X.equals(self.X) and not y.equals(self.y): raise ValueError('"pre_split" should be used on raw X and y data.')
            self.pre_split = pre_split
            self.X, self.X_test, self.y, self.y_test = X_train, X_test, y_train, y_test
      
        return X_train, X_test, y_train, y_test
    
    

    def _define_kf(self, kfold=None, n_splits=None, random_state=None):
        random_state = self.random_state if random_state is None else random_state
        kf = None
        if (kfold is None) and (n_splits is not None):
            # If target is continoous, use Kfold for cross validation
            if self.is_y_continoous:
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            # If target is categorical, use StratifiedKFold
            else:
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        elif kfold is not None:
            kf= kfold
        
        if kf is not None:
            return kf, kf.n_splits 
        else :
            return None, None
    

    def run_pipeline(self, scoring, 
                     X=None, y=None,
                     test_size=0.2, n_splits=None, kf=None, n_jobs_cv=-1, 
                     use_mlflow=False, run_name=None, use_mlflow_cv=None, run_name_cv=None):
        
        """
        Execute a machine learning pipeline including data preprocessing, model training, and evaluation.

        Parameters:
            - scoring (str): The scoring metric used for model evaluation.
            - X (df, optionnal, default=None): Run pipeline with X, rather than self.X
            - y (df, optionnal, default=None): Run pipeline with y, rather than self.y
            - test_size (float, default=0.2): The proportion of the dataset to include in the test split.
            - n_splits (int, default=None): Number of folds in cross-validation (KFold/StratifiedKFold).
            - kf (KFold, default=None): Predefined cross-validation splitter. If provided, n_splits is ignored.
            - n_jobs_cv (int, default=-1): Number of CPU cores to use for cross-validation.
            - use_mlflow (bool, default=False): Whether to use MLflow for experiment tracking.
            - run_name (str, default=None): Name of the MLflow run.
            - use_mlflow_cv (bool, default=None): Whether to use MLflow for cross-validation.
            - run_name_cv (str, default=None): Name of the MLflow run for cross-validation.

        Returns:
            If use_mlflow is True, returns None. Otherwise, returns a tuple containing train_score, test_score, signature, and an example of preprocessed training data.

        Note:
            This function executes a machine learning pipeline, including data preprocessing, model training, and evaluation. If use_mlflow is True, it tracks the experiment using MLflow.
        """
        def core_run_pipeline():
            # Separate data into training and validation sets
            X_train, X_test, y_train, y_test = self._split_data(X, y, test_size)

            if kf is not None:
                _ = self._run_cross_validation(X_train, y_train, kf, scoring, n_jobs=n_jobs_cv, use_mlflow=use_mlflow_cv, run_name=run_name_cv)
            
            for (name, pipeline) in self.data_pipelines_:
                X_train, X_test = self.apply_data_pipeline((name, pipeline), X_fit_tr=X_train, X_transform=X_test)

            # To handle sampling transformation (e.g outlier removal)
            y_train = self.remove_dropped_index_in_y(X_train, y_train)
            y_test = self.remove_dropped_index_in_y(X_test, y_test)

            print(f'\n{Fore.MAGENTA}{"~"*10}\n{Fore.GREEN}Training on X_train data...')
            # Fit the pipeline on the training data
            self.model_.fit(X_train, y_train) 
            # Make predictions on the test data
            predictions = self.model_.predict(X_test)
            # Get the scores
            train_score = self.model_.score(X_train, y_train)
            test_score = self.evaluate_model_get_score(scoring, y_test, predictions)
            
            signature = infer_signature(X_train, predictions)
            
            print(f"Performance on train data: {Fore.GREEN}{scoring} {Fore.WHITE}= {Fore.RED}{train_score}")
            print(f"Performance on test data: {Fore.GREEN}{scoring} {Fore.WHITE}= {Fore.RED}{test_score}") 
            
            return (train_score, test_score, signature, X_train.head()) if use_mlflow else (train_score, test_score)


        X, y = (self.X, self.y) if X is None and y is None else (X, y)
        
        kf, _ = self._define_kf(kfold=kf, n_splits=n_splits) 

        # Run cross-validation on the whole dataset and then return
        if (not test_size) and (kf is not None):
            _ = self._run_cross_validation(X, y, kf, scoring, n_jobs=n_jobs_cv, use_mlflow=use_mlflow_cv, run_name=run_name_cv)   
            return 


        if use_mlflow:
            # MLflow-related code
            experiment_id = get_or_create_mlflow_experiment(run_name)
            mlflow.set_experiment(experiment_id=experiment_id)
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
                train_score, test_score, signature, X_train_exemple = core_run_pipeline()

                for name_pipeline, pipeline in self.data_pipelines_:
                    mlflow.sklearn.log_model(pipeline, name_pipeline)
                
                
                from sklearn.base import BaseEstimator
                if isinstance(self.model_, BaseEstimator):
                    mlflow.sklearn.log_model(self.model_, self.model_name_, signature=signature, input_example=X_train_exemple)
                else:
                    from MLp.src.secondary_modules.mlflow_functions import CustomModelForMlflow
                    mlflow.pyfunc.log_model(artifact_path="model", python_model=CustomModelForMlflow(self.model_, self.data_pipelines_, self.apply_created_data_pipelines), 
                                            signature=signature, input_example=X_train_exemple)

                mlflow.log_params(self._get_hyperparameters())
                mlflow.log_metrics({''.join([scoring, '_', 'test_score']): np.mean(test_score),
                                    "std_test_score": np.std(test_score),
                                    ''.join([scoring, '_', 'train_score']): np.mean(train_score),
                                    "std_train_score": np.std(train_score)
                                    })  

        else:
            core_run_pipeline()  
        return 


    # Method that runs the preprocessing pipeline and fit the model with a crossvalidation approach
    def _run_cross_validation(self, X, y, kf, scoring, n_jobs=-1, use_mlflow=False, run_name=None, verbose=True):
        """
        Run cross-validation for the model training.

        Parameters:
            X (DataFrame): Input features.
            y (Series): Target variable.
            kf (KFold or StratifiedKFold): Cross-validation splitter.
            scoring (str): Scoring metric for evaluation.
            n_jobs (int): Number of parallel jobs for cross-validation.
            use_mlflow (bool): Flag indicating whether to use MLflow for experiment tracking.
            run_name (str): Name of the MLflow run.
            verbose (bool): Flag indicating whether to print progress messages.

        Returns:
            list: Test scores for each fold.
        """

        # Nested function to fit and predict on a single fold
        def _cv_train_and_predict(counter, train_index, test_index, X_cv, y_cv, scoring):
            def _core_cv_train_and_predict():
                # Get X/y train/test according to the kf data slicing
                X_train_cv, y_train_cv = X_cv.iloc[train_index], y_cv.iloc[train_index]
                X_valid_cv, y_valid_cv = X_cv.iloc[test_index], y_cv.iloc[test_index]

                for (name, pipeline) in self.data_pipelines_:
                    X_train_cv, X_valid_cv = self.apply_data_pipeline((name, pipeline), X_fit_tr=X_train_cv, X_transform=X_valid_cv)
                y_train_cv = self.remove_dropped_index_in_y(X_train_cv, y_train_cv)
                y_valid_cv = self.remove_dropped_index_in_y(X_valid_cv, y_valid_cv)

                # Fit the pipeline on the training data
                self.model_.fit(X_train_cv, y_train_cv)
                # Make predictions on the test data
                predictions = self.model_.predict(X_valid_cv)
                # Get the scores
                train_score = self.model_.score(X_train_cv, y_train_cv)
                test_score = self.evaluate_model_get_score(scoring, y_valid_cv, predictions)
                return (train_score, test_score)
            
            if use_mlflow:
                with mlflow.start_run(experiment_id=experiment_id, run_name=''.join([run_name, '_', str(counter)]), nested=True):      
                    (train_score, test_score) = _core_cv_train_and_predict()
                    mlflow.log_params(self._get_hyperparameters())
                    mlflow.log_metric(''.join([scoring, '_', 'cv_score']), test_score)
                    
            else:
                (train_score, test_score) = _core_cv_train_and_predict()
            
            return (train_score, test_score)
                    
        def core_cv():
            # Parallelize the training and prediction process
            train_test_scores = Parallel(n_jobs=n_jobs)(
                delayed(_cv_train_and_predict)(
                    counter, train_index, test_index, X, y, scoring
                ) for counter, (train_index, test_index) in enumerate(kf.split(X, y))
            )
            train_cv_scores = [scores[0] for scores in train_test_scores]
            test_cv_scores = [scores[1] for scores in train_test_scores]
            return train_cv_scores, test_cv_scores
        
        if verbose: print(f'\n{Fore.MAGENTA}{"~"*10}\n{Fore.GREEN}Cross-validation is being performed...')
         
        if use_mlflow:
            # MLflow-related code
            experiment_id = get_or_create_mlflow_experiment(run_name)
            mlflow.set_experiment(experiment_id=experiment_id)
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
                train_cv_scores, test_cv_scores = core_cv()
                mlflow.log_params(self._get_hyperparameters())
                mlflow.log_metric(''.join([scoring, '_', 'cv_score']), np.mean(test_cv_scores))
                mlflow.log_metric("std_cv_score", np.std(test_cv_scores))
                
        else:
            train_cv_scores, test_cv_scores = core_cv()
        
        if verbose:
            print(f'Mean train scores for cross_valid {Fore.GREEN}{scoring} {Fore.WHITE}= {Fore.RED}{np.mean(train_cv_scores)} {Fore.MAGENTA}--- {Fore.WHITE}Standart deviation = {Fore.RED}{np.std(train_cv_scores)}')
            print(f'Mean test scores for cross_valid {Fore.GREEN}{scoring} {Fore.WHITE}= {Fore.RED}{np.mean(test_cv_scores)} {Fore.MAGENTA}--- {Fore.WHITE}Standart deviation = {Fore.RED}{np.std(test_cv_scores)}')
             
        return test_cv_scores
    
    
    

    def _get_hyperparameters(self):
        """
        Retrieve hyperparameters from data pipelines and the model.

        Returns:
            dict: A dictionary containing hyperparameters.
        """
        my_hyperparams = {}
        for pipeline_name, pipeline in self.data_pipelines_:
            pipeline_params = pipeline.get_params()
            my_hyperparams.update({''.join([pipeline_name, '__', name]):value for name, value in pipeline_params.items()})
        my_hyperparams.update({''.join([self.model_name_, '__', name]):value for name, value in self.model_.get_params().items()})
        return my_hyperparams


    def best_hyperparams_fit(self, scoring, train_size, verbose=True):
        """
        Refits created pipelines and the model with the best hyperparameters for a given scoring metric and train size.

        Parameters:
        - scoring: str, the scoring metric used to determine the best hyperparameters.
        - train_size: str, the size of the training set used to determine the best hyperparameters.
        - verbose: bool, whether to print verbose output (default is True).

        Returns:
        - None
        """
        best_hyperparams = self.best_hyperparameters[scoring][train_size]["best_params_separate_objet"]
        
        if verbose: print(f'{Fore.GREEN}Running best_hyperparams_fit...\n{Fore.WHITE}Refit all created pipelines and the model on:')
        
        for pipeline_name, pipeline in self.data_pipelines_:
            if verbose: print(f'{Fore.GREEN}{pipeline_name}: {Fore.BLUE}{best_hyperparams[pipeline_name]}')
            pipeline.set_params(**best_hyperparams[pipeline_name])
        
        self.model_.set_params(**best_hyperparams[self.model_name_])
        if verbose: print(f'{Fore.GREEN}{self.model_name_}: {Fore.BLUE}{best_hyperparams[self.model_name_]}')
        return



  #  @valid_mlp.validate_grid_search
    def grid_search_tuning(self, param_grid: Dict, scoring: str, 
                                 X=None, y=None, kf=None, n_splits=5, cv_n_jobs=-1, 
                                 use_mlflow=False, run_name=None, 
                                 verbose=True):

        """
        Perform hyperparameter tuning using grid search technique.
    
        Parameters:
            param_grid (Dict): A dictionary specifying the hyperparameters to be tuned.
            scoring (str): The scoring metric to be used for evaluating the models.
            X (df, optional): Feature dataset. Defaults to None (self.X used).
            y (df, optional): Target labels. Defaults to None (self.y used).
            kf (object): Object representing the cross-validation strategy. Defaults to None. Overwritte n_split if provided.
            n_splits (int): Number of splits in cross-validation. Defaults to 5.
            cv_n_jobs (int): Number of jobs to run in parallel for cross-validation. Defaults to -1.
            use_mlflow (bool): Flag indicating whether to use MLflow for experiment tracking. Defaults to False.
            run_name (str): Name of the MLflow experiment run. Defaults to None.
            verbose (bool): Flag indicating whether to print progress messages. Defaults to True.
    
        Returns:
            Tuple: A tuple containing the best hyperparameters found, the corresponding cross-validated score, 
                   and the standard deviation of the cross-validated scores.
        """
         

        def _update_pipelines_or_model_hyperparameters():
            """
            Update pipelines with hyperparameters specified in param_grid.
            Note: Assumes hyperparameter names are formatted as 'objet_name__param_name'.
            """
            for name, param_value in dict_trial.items():
                objet_name, param_name = name.split('__', 1)

                if objet_name == self.model_name_:
                    self.model_.set_params(**{param_name: param_value})
                else:
                    for created_pipe_name, pipeline in self.data_pipelines_:
                        if objet_name == created_pipe_name:
                            pipeline.set_params(**{param_name: param_value})
            return
    
            
        def _update_best(b_params, b_cv_score, b_score_std, params, scores_cv):
            # Update best_score and best_params if a better combination is found
            if b_cv_score is None or np.mean(scores_cv) > b_cv_score:
                best_p = params
                best_cv_s = np.mean(scores_cv)
                best_std = np.std(scores_cv)
                return best_p, best_cv_s, best_std
            else:
                return b_params, b_cv_score, b_score_std
    
        kf, n_splits = self._define_kf(kfold=kf, n_splits=n_splits) 
        
        X_data, y_data = (self.X, self.y) if X is None and y is None else (X, y)

        if verbose: print(f'\n{Fore.MAGENTA}{"~"*10}\n{Fore.GREEN}Running grid_search optimization...')     

        best_params, best_cv_score, best_score_std = None, None, None
        # Generate all possible combinations of hyperparameters
        param_combinations = ParameterGrid(param_grid) 
            
        if use_mlflow:
            # MLflow-related code
            experiment_id = get_or_create_mlflow_experiment(run_name)
            mlflow.set_experiment(experiment_id=experiment_id)
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
                
                for count, dict_trial in enumerate(param_combinations):
                    with mlflow.start_run(run_name=''.join([run_name, '_', str(count)]), nested=True):
                        _update_pipelines_or_model_hyperparameters()
                        # Fit the grid search to the data
                        cv_scores = self._run_cross_validation(X_data, y_data, kf, scoring, n_jobs=cv_n_jobs, verbose=False)
                        # Log to MLflow
                        mlflow.log_params(dict_trial)
                        mlflow.log_metric(''.join([scoring, '_', 'cv_score']), np.mean(cv_scores))
                        mlflow.log_metric("std_cv_score", np.std(cv_scores))
                        #Update best variables
                        best_params, best_cv_score, best_score_std = _update_best(best_params, best_cv_score, best_score_std, dict_trial, cv_scores)
                mlflow.log_params(best_params)
                mlflow.log_metric(''.join([scoring, '_', 'cv_score']), best_cv_score)
                mlflow.log_metric("std_cv_score", best_score_std)
                
        else:

            for count, dict_trial in enumerate(param_combinations):
                _update_pipelines_or_model_hyperparameters()
                # Fit the grid search to the data
                cv_scores = self._run_cross_validation(X_data, y_data, kf, scoring, n_jobs=cv_n_jobs, verbose=False)
                best_params, best_cv_score, best_score_std = _update_best(best_params, best_cv_score, best_score_std, dict_trial, cv_scores)
        
        if verbose:
            print(f'Grid search:\nBest cross-valid {Fore.GREEN}{scoring} {Fore.WHITE}test score = {Fore.RED}{best_cv_score}')
            print(f'{Fore.WHITE}Best score standart deviation = {Fore.RED}{best_score_std}\n{Fore.WHITE}Best hyperparameters: {Fore.BLUE}{best_params}')
            
        self._actualize_grid_attributes_if_best(scoring, None, n_splits, best_params, best_cv_score, best_score_std)        
        return best_params, best_cv_score, best_score_std



    @valid_mlp.validate_optuna
    def optuna_tuning(self, optuna_hyperparameters: List, n_trials: int, scoring: str, 
                             direction='maximize', custom_objective=None,
                             X=None, y=None, n_splits=None, kf=None, cv_n_jobs=-1,
                             use_mlflow=False, run_name=False,
                             verbose=True):
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            optuna_hyperparameters (List): List of tuples specifying hyperparameters to optimize.
                                           Each tuple should contain (param_name, param_type, param_range, param_options).
            n_trials (int): Number of trials for hyperparameter optimization.
            scoring (str): Scoring metric to optimize.
            direction (str, optional): Direction to optimize ('maximize' or 'minimize'). Default is 'maximize'.
            custom_objective (function, optional): Custom objective function for optimization. Default is None.
            X (df, optional): Feature dataset. Defaults to None (self.X used).
            y (df, optional): Target labels. Defaults to None (self.y used).
            n_splits (int, optional): Number of splits for cross-validation. Default is None.
            kf (str or object, optional): Cross-validation strategy. Default is None.
            cv_n_jobs (int, optional): Number of jobs for cross-validation. Default is -1.
            use_mlflow (bool, optional): Flag to enable MLflow logging. Default is False.
            run_name (str, optional): Name for the MLflow run. Default is False.
            verbose (bool, optional): Flag to display progress messages. Default is True.

        Returns:
            Tuple: Best hyperparameters, best cross-validation score, and standard deviation of the score.
        """
        def objective(trial):
            #Objective function for Optuna optimization.
            def _core_objective():
                #Core objective function to set hyperparameters and calculate cross-validation score.
                def _create_optuna_dict():
                    #Create dictionary of hyperparameters to optimize.
                    optuna_dict = {name: [] for name, _ in self.data_pipelines_}
                    optuna_dict[self.model_name_] = []

                    for param_name, param_type, param_range, param_options in optuna_hyperparameters:
                        name_objet, name_params = param_name.split('__', 1)

                        if optuna_dict.get(name_objet, None) is not None:
                            optuna_dict[name_objet].append((name_params, param_type, param_range, param_options))

                    return optuna_dict

                def _set_pipeline_params(name, pipeline_or_model, params, trial) -> Dict:
                    #Set hyperparameters for a pipeline/model.
                    parameters = {}
                    for param_name, param_type, param_range, param_options in params:
                        trial_value = getattr(trial, param_type)(''.join([name, '__', param_name]), *param_range, **param_options)
                        pipeline_or_model.set_params(**{param_name: trial_value})
                        parameters[''.join([name, '__', param_name])] = trial_value
                    return parameters
            
                optuna_dict = _create_optuna_dict()
                optuna_trial_params = {}
                for pipeline_name, pipeline in self.data_pipelines_:
                    if optuna_dict.get(pipeline_name, None) is not None:
                        parameters = _set_pipeline_params(pipeline_name, pipeline, optuna_dict[pipeline_name], trial)
                        optuna_trial_params.update(parameters)
                if optuna_dict.get(self.model_name_, None) is not None:
                    parameters = _set_pipeline_params(self.model_name_, self.model_, optuna_dict[self.model_name_], trial)  
                    optuna_trial_params.update(parameters) 
                # Use cross-validation to get the objective score
                cv_scores = self._run_cross_validation(X, y, kf, scoring, n_jobs=cv_n_jobs, verbose=False)
                # Set the standard deviation as a user attribute
                trial.set_user_attr('std_cv_score', np.std(cv_scores))
                
                return optuna_trial_params, cv_scores

            if use_mlflow:
                with mlflow.start_run(run_name=''.join([run_name, '_', str(trial.number)]), nested=True):
                    optuna_trial_params, cv_scores = _core_objective()
                    # Log to MLflow
                    mlflow.log_params(optuna_trial_params)
                    mlflow.log_metric(''.join([scoring, '_', 'cv_score']), np.mean(cv_scores))
                    mlflow.log_metric("std_cv_score", np.std(cv_scores))
            else:
                optuna_trial_params, cv_scores = _core_objective()
                
            return np.mean(cv_scores)


        def optuna_optimization():
            def _new_best_value_callback(study, frozen_trial):
                """
                Logging callback that will report when a new trial iteration improves upon existing
                best trial values.
                """
                best = study.user_attrs.get("best", None)
                if study.best_value and best != study.best_value:
                    study.set_user_attr("best", study.best_value)
                    if best:
                        improvement_percent = (abs(best - study.best_value) / study.best_value) * 100
                        print(
                            f"{Fore.WHITE}Trial {frozen_trial.number} (new best trial) achieved value: {Fore.RED}{frozen_trial.value} {Fore.WHITE}with"
                            f"{Fore.RED}{improvement_percent: .4f}% {Fore.WHITE}improvement compared to previous best trial."
                        )
                    else:
                        print(f"{Fore.WHITE}Initial trial {frozen_trial.number} achieved value: {Fore.RED}{frozen_trial.value}")

                    
            if direction.lower() not in ['maximize', 'minimize']:
                raise ValueError('To run optuna tuning, please set either "minimize" or "maximize" to direction.')
            
            optuna_tuning = optuna.create_study(direction=direction)
            if custom_objective is not None:
                optuna_tuning.optimize(custom_objective, n_trials=n_trials, callbacks=[_new_best_value_callback]) 
            else:
                optuna_tuning.optimize(objective, n_trials=n_trials, callbacks=[_new_best_value_callback]) 
            return optuna_tuning


        def get_best_hyperparameters(optuna_object):
            best_params = optuna_object.best_params
            best_score = optuna_object.best_value
            best_trial = optuna_object.best_trial
            best_score_std = best_trial.user_attrs.get('std_cv_score', None)
            return best_params, best_score, best_score_std
        
        
        X, y = (self.X, self.y) if X is None and y is None else (X, y) 

        # Choose the appropriate cross-validation strategy based on the target type
        kf, n_splits = self._define_kf(kfold=kf, n_splits=n_splits)
        

        if verbose: print(f'\n{Fore.MAGENTA}{"~"*10}\n{Fore.GREEN}Running Optuna optimization...')  

        if use_mlflow:
            # MLflow-related code
            experiment_id = get_or_create_mlflow_experiment(run_name)
            mlflow.set_experiment(experiment_id=experiment_id)

            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
                optuna_opti = optuna_optimization()
                best_params, best_score, best_score_std = get_best_hyperparameters(optuna_opti) 

                mlflow.log_params(best_params)
                mlflow.log_metric(''.join([scoring, '_', 'cv_score']), best_score)
                mlflow.log_metric("std_cv_score", best_score_std)

        else:
            # Code to execute when MLflow is not used
            optuna_opti = optuna_optimization()
            best_params, best_score, best_score_std = get_best_hyperparameters(optuna_opti) 


        if verbose:
            print(f'\n{Fore.WHITE}Optuna:\nBest cross-valid {Fore.GREEN}{scoring} {Fore.WHITE}test score = {Fore.RED}{best_score}')
            print(f'{Fore.WHITE}Best score standard deviation = {Fore.RED}{best_score_std}\n{Fore.WHITE}Best hyperparameters: {Fore.BLUE}{best_params}')

        self._actualize_grid_attributes_if_best(scoring, None, n_splits, best_params, best_score, best_score_std)

        return best_params, best_score, best_score_std



        
    def make_hyperparameters_dict_with_separated_objects(self, hyperparameters):
        hyperparameter_dict = {name: {} for name, _ in self.data_pipelines_}
        hyperparameter_dict[self.model_name_] = {}
        
        for param_name, best_value in hyperparameters.items():
            name_objet, name_params = param_name.split('__', 1)
            if hyperparameter_dict.get(name_objet, None) is not None:
                hyperparameter_dict[name_objet][name_params] = best_value
        return hyperparameter_dict   

    def _actualize_grid_attributes_if_best(self, scoring, test_size, n_splits, best_params, best_score, best_std):
        def _assess_data_size(test_size=None, n_splits=None):
            """
            Assess the size of the data based on different scenarios.
            Parameters:
            - pre_split: Percentage of data pre-split. None if not pre-split.
            - test_size: Percentage of data used for testing in train_test_split. None if not used.
            - n_splits: Number of splits in cross-validation. None if not used.
            Returns:
            - size: The assessed size of the data.
            """
            # Initial size
            train_size = 1
            # Adjust size based on pre-split
            if self.pre_split is not None and self.pre_split != 0:
                train_size *= 1 - self.pre_split
            # Adjust size based on train_test_split
            if test_size is not None and test_size != 0:
                train_size *= 1 - test_size
            # Adjust size based on cross-validation
            if n_splits is not None and n_splits != 0:
                valid_size = 1 / n_splits
                train_size *= 1 - valid_size
            return round(train_size, 1)
    
            
        size = _assess_data_size(test_size=test_size, n_splits=n_splits)
        hyperparam_dict = self.make_hyperparameters_dict_with_separated_objects(best_params)
        # Actualize values if there is no value OR saved best_score_ is smaller than best_score
        if not self.best_hyperparameters.get(scoring) or not self.best_hyperparameters[scoring].get(size) or self.best_hyperparameters[scoring][size]['best_score'] < best_score:
            self.best_hyperparameters.setdefault(scoring, {size:{}})
            self.best_hyperparameters[scoring][size] = {'best_params_separate_objet': hyperparam_dict,
                                                        'best_params_all': best_params,
                                                        'best_score': best_score,
                                                        'best_std': best_std,
                                                    }

            print(f'\n\n{Fore.YELLOW}NEW BEST HYPERPARAMETER COMBINATION IDENTIFIED:')
            
            print(f'{Fore.WHITE}Scoring method: {Fore.GREEN}{scoring}{Fore.WHITE}\tTraining dataset size: {Fore.RED}{size}')
            print(f'{Fore.WHITE}Best {Fore.GREEN}{scoring}{Fore.WHITE} score: {Fore.RED}{self.best_hyperparameters[scoring][size]["best_score"]}')
            print(f'{Fore.WHITE}Best {Fore.GREEN}{scoring}{Fore.WHITE} score std: {Fore.RED}{self.best_hyperparameters[scoring][size]["best_std"]}')
            print(f'{Fore.WHITE}Best hyperparameters per pipeline/model:')
            for objet in self.best_hyperparameters[scoring][size]['best_params_separate_objet']:
                print(f'{Fore.WHITE}- {Fore.GREEN}{objet}:\n\t{Fore.BLUE}{self.best_hyperparameters[scoring][size]["best_params_separate_objet"][objet]}')
            print(f'{Fore.WHITE}All best hyperparameters:\n{Fore.BLUE}{self.best_hyperparameters[scoring][size]["best_params_all"]}\n\n')
                
        return       




    # Method that enables to train the assembled model on all our data
    def train_on_all_data(self, X=None, y=None, 
                          get_pipelines_model=True, verbose=None, 
                          use_mlflow=None, run_name='on_all_data'):
        """
        Fit all X data on sampling_pipeline, preprocessing_pipeline and the model. Potentially perform predictions based on unseen data.
        
        Parameters:
        - scoring: scoring method to be used to assess model's performance. 
        - predict: take either 'pre_split' or 'to_predict' as value. 
                   Indicate either the model should predict based on pre_split data or on to_predict data (provided by running the 'initialize_data' method) 
        
        Returns:

        """
        
        def core_train_on_all_data(X_data, y_data):
            X, y = [self.X, self.y] if X_data is None and y_data is None else [X_data, y_data]
            # Apply the data pipelines
            for (name, pipeline) in self.data_pipelines_:
                X = self.apply_data_pipeline((name, pipeline), X_fit_tr=X)
            y = self.remove_dropped_index_in_y(X, y)

            self.model_.fit(X, y)
            print(f'{Fore.RED}The pipelines and model have been fitted on X.')
            return X, y if use_mlflow else None
        
        if verbose: print(f'\n{Fore.MAGENTA}{"~"*10}\n{Fore.GREEN}Running train_on_all_data...')  
        
        if use_mlflow:
            # MLflow-related code
            experiment_id = get_or_create_mlflow_experiment(run_name)
            mlflow.set_experiment(experiment_id=experiment_id)
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
                X, y = core_train_on_all_data(X, y)
                
                for name_pipeline, pipeline in self.data_pipelines_:
                    mlflow.sklearn.log_model(pipeline, name_pipeline)

                if isinstance(self.model_, BaseEstimator):
                    mlflow.sklearn.log_model(self.model_, self.model_name_, input_example=X.head())
                else:
                    from MLp.src.secondary_modules.mlflow_functions import CustomModelForMlflow
                    mlflow.pyfunc.log_model(artifact_path="model", python_model=CustomModelForMlflow(self.model_, self.data_pipelines_, self.apply_created_data_pipelines), 
                                            input_example=X.head())
                mlflow.log_params(self._get_hyperparameters())
                
        else:
            core_train_on_all_data(X, y)                 
                
        return (self.data_pipelines_, (self.model_name_, self.model_)) if get_pipelines_model else None



    def save_pipelines_model(self, dir_path, signature, serialization_format='cloudpickle', metadata=None, custom_save=None, custom_save_params=None):
        if callable(custom_save) and custom_save_params is not None:
            return custom_save(**custom_save_params)
        
        my_data_pipelines = self.data_pipelines_
        my_model = (self.model_name_, self.model_)

        if my_data_pipelines is not None and my_data_pipelines:
            for pipeline_name, pipeline in my_data_pipelines:
                path = os.path.join(dir_path, pipeline_name)
                mlflow.sklearn.save_model(
                    pipeline,
                    path,
                    serialization_format=serialization_format,
                    signature=signature,
                    metadata=metadata
                )
                
        if my_model is not None and my_model:
            path = os.path.join(dir_path, my_model[0])
            if isinstance(my_model[1], BaseEstimator):
                mlflow.sklearn.save_model(
                    my_model[1],
                    path,
                    serialization_format=serialization_format,
                    signature=signature,
                    metadata=metadata
                )     
        return


    def feature_importance(self, method, X=None, y=None, scoring=''):
        print(' ------------------', '\n', 'Computing features importance', '\n', '------------------')
        X, y = (self.X, self.y) if X is None and y is None else (X, y)
        
        for (name, pipeline) in self.data_pipelines_:
            X = self.apply_data_pipeline((name, pipeline), X_fit_tr=X)
        y = self.remove_dropped_index_in_y(X, y)
        
        self.model_.fit(X, y)
        self.check_feature_importance(method=method, scoring=scoring, X=X, y=y)
        return
    

    # Method that enables to use the assembled model to make real life predictions on unseen data 
    def make_test_inferences(self, predict, scoring, model=None, data_pipelines=None):
        
        data_pipelines = self.data_pipelines_ if data_pipelines is None else data_pipelines
        model = self.model_ if model is None else model
        
        if predict == 'pre_split':
            # Read data to predict
            X_to_predict = self.X_test
            y_to_predict = self.y_test

            for (name, pipeline) in data_pipelines:
                X_to_predict = self.apply_data_pipeline((name, pipeline), X_transform=X_to_predict)
            y_to_predict = self.remove_dropped_index_in_y(X_to_predict, y_to_predict)

            predictions = model.predict(X_to_predict)
            score = self.evaluate_model_get_score(scoring, y_to_predict, predictions)
            print(f"Predictions with pre-splitted data: {scoring} = {score}")
            
            return score
        
        else:
            for (name, pipeline) in data_pipelines:
                X_to_predict = self.apply_data_pipeline((name, pipeline), X_transform=X_to_predict)            
            
            predictions = model.predict(X_to_predict)
            
            return predictions
















