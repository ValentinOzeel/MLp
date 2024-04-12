import os
import sys
import types

from MLp.src.preprocessing.transformer_decorators import get_num_cat_features_decorator, select_columns_decorator
import MLp.src.secondary_modules.validate_args_kwargs as valid_mlp
from MLp.src.secondary_modules.import_libraries import import_cpu_gpu_sklearn, import_cpu_gpu_pandas
pd = import_cpu_gpu_pandas()
    
Pipeline = import_cpu_gpu_sklearn('pipeline', 'Pipeline')
BaseEstimator, TransformerMixin = import_cpu_gpu_sklearn('base', ['BaseEstimator', 'TransformerMixin'])


def get_cpu_count():
    try:
        # If on Linux or MacOS
        if "linux" in sys.platform or "darwin" in sys.platform:
            return os.cpu_count()
        # If on Windows
        elif "win" in sys.platform:
            return int(os.environ["NUMBER_OF_PROCESSORS"])
        else:
            return None
    except Exception as e:
        print(f"Error while determining CPU count: {e}")
        return None
    
os.environ["OMP_NUM_THREADS"] = str(get_cpu_count())


############################## PREPROCESSING TRANSFORMERS ##############################
############################## PREPROCESSING TRANSFORMERS ##############################

class DynamicFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer class that aims at adding flexibility to a given sklearn transformer. 
    It applies the transformer to selected columns of a DataFrame, enables preprocessing before transformation 
    and add the new feature while optionally removing initial features (add add prefix or rename transformed features).

    Parameters:
    -----------
    transformer : object
        The transformer to apply to the selected columns.
    _preprocess_transformer : object, optional, default=None
        Preprocess transformer to apply before applying the main transformer.
    _preprocess_params : dict, optional, default={}
        Parameters for the preprocess transformer.
    _columns : list or array-like, optional, default=None
        List of columns to apply the transformer on. If None, applies to all columns.
    _drop : bool, optional, default=False
        Whether to drop the selected columns after transformation.
    _col_name_prefix : str, optional, default=None
        Prefix to use for column names of the transformed features.
    _col_names : list of str, optional, default=None
        List of names for the transformed features. Overrides _col_name_prefix.
    **kwargs : dict
        All keyword arguments to pass to the transformer initialization.

    Attributes:
    selected_cols : list
        List of columns selected for transformation.

    Methods:
    fit(X, y=None)
        Fit the transformer on the input data.
    transform(X)
        Transform the input data using the fitted transformer.
    """
    def __init__(self, transformer,
                 _preprocess_transformer=None,
                 _preprocess_params={},
                 _columns=None, _drop=False, 
                 _col_name_prefix=None, _col_names=None,
                 **kwargs
                 ):
        
        self.transformer = transformer(**kwargs)
         
        self._preprocess_transformer = _preprocess_transformer() if _preprocess_transformer is not None else None
        self._preprocess_params = _preprocess_params if _preprocess_params else {}
        if self._preprocess_transformer is not None and self._preprocess_params:
            self._preprocess_transformer.set_params(**self._preprocess_params)
        self._columns = _columns
        self._drop = _drop
        self._col_name_prefix = _col_name_prefix
        self._col_names = _col_names
        
        self.selected_cols = []

    # Decorator to select the columns to apply the transformer on
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        if self._preprocess_transformer:
            # Get process transformer, preprocessed df and preprocessed cols
            X_preprocessed, self._preprocess_transformer = self._get_preprocess_transform(X[self.selected_cols], self._preprocess_transformer)
            # Fit the model only with the preprocessed columns
            self.transformer.fit(X_preprocessed)
        else:    
            self.transformer.fit(X[self.selected_cols])
        return self

    def transform(self, X):
        self._validate_parameters()
        
        if self._preprocess_transformer:
            X_preprocessed = pd.DataFrame(self._preprocess_transformer.transform(X[self.selected_cols]), index=X.index)
            new_df = pd.DataFrame(self.transformer.transform(X_preprocessed), index=X.index)
        else: 
            new_df = pd.DataFrame(self.transformer.transform(X[self.selected_cols]), index=X.index)
            
        col_names = self._get_col_names(new_df)
        new_df.columns = col_names

        if self._drop: X = X.drop(columns=self.selected_cols)

        return pd.concat([X, new_df], axis=1)

    def _get_preprocess_transform (X, preprocess_transformer):
        # Get the full preprocessed df with preprocessed columns and non preprocessed columns
        X_preprocessed = preprocess_transformer.fit_transform(X)
        # Reassign indexing lost during transofrmation
        X_preprocessed.index = X.index
        return X_preprocessed, preprocess_transformer

    def _validate_parameters(self):
        if self._drop is not None and (self._col_name_prefix is None and self._col_names is None):
            raise ValueError('Need to provide "_col_name_prefix" or "_col_names" parameter if "_drop" is selected.')
        
    def _get_col_names(self, df):
        if self._col_names and len(self._col_names) == len(self.selected_cols):
            return self._col_names
        elif self._col_name_prefix and isinstance(self._col_name_prefix, str):
            return [f"{self._col_name_prefix}{i+1}" for i in range(df.shape[1])]
        else:      
            return self.selected_cols
        
        
class FunctionTransformerTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies a function transformer to specific columns of a DataFrame.
    
    Parameters:
    -----------
    function_transformer : function
        The function transformer to be applied to the selected columns.
    
    columns : list or None, optional (default=None)
        The list of column names to apply the function transformer to. If None, the function transformer 
        will be applied to all columns.
    
    drop : bool or list, optional (default=False)
        Whether to drop the original columns after transformation. If True, drops the transformed columns.
        If a list is provided, drops the specified columns after transformation.
    """
    def __init__(self, 
                 function_transformer,
                 columns=None,
                 drop=False):
        
        self.function_transformer = function_transformer
        self.columns = columns
        self.drop = drop

        self.selected_cols = []

    # Decorator to select the columns to apply the transformer on
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result_function_transformer = self.function_transformer(X[self.selected_cols])
        
        if self.drop:
            self.drop = self.drop if isinstance(self.drop, list) else self.selected_cols
            X = X.drop(columns=self.drop)

        return pd.concat([X, result_function_transformer], axis=1)
            

    ############################## END OF PREPROCESSING TRANSFORMERS ##############################
    ############################## END OF PREPROCESSING TRANSFORMERS ##############################


   

class MLpPreprocessing:
    '''
    This class provides a structured and modular approach to data preprocessing, allowing users to define and apply preprocessing pipelines efficiently.
    '''
    def __init__(self):
        # Created pipelines
        self.data_pipelines_ = []

    def create_data_pipeline(self, name: str, steps=None, pipeline=None)->Pipeline:   
        """
        Create or extend a scikit-learn pipeline for data preprocessing.

        Parameters:
        - name (str): Name of the pipeline. (should include 'sampling' if there is any sampling performed, such as outlier removal)
        - steps (list, optional): List of tuples. Each tuple contains 3 or 4 elements:
            - a step name (If starts with '*', the caracter will by removed and the transformer will be used in its native form as passed. 
                           Else, the transformer will be wrapped by DynamicFeatureTransformer)
            - transformer or function transformer: can be any transformer or function transformer (NOT its instance)
            - transformer's parameters: Dict of transformer's parameters
            - optional dict of additional parameters for transformer augmentation class DynamicFeatureTransformer
        - pipeline (Pipeline, optional): Existing pipeline to which steps will be added.
        Either steps or pipeline must be passed

        Returns:
        - new_pipeline (Pipeline): Scikit-learn pipeline created or extended based on the provided steps or pipeline.
        """

        def _add_counter_step_names(step_list):
            """
            Add numerical suffixes to step names that already exist to ensure uniqueness in the pipeline.

            Parameters:
            - step_list (list): List of dictionaries representing steps.

            Returns:
            - new_step_list (list): List of steps with modified names.
            """
        
            def _generate_modified_step(step, count):
                return "-".join([step, str(count)])
            def _generate_unique_name(step, unique_names):
                step_name = step[0]            
                if step_name not in unique_names:
                    unique_names.add(step_name)
                    step = (step_name,) + step[1:]
                    return step, unique_names
                else:
                    count = 2
                    modified_step_name = _generate_modified_step(step_name, count)
                    while modified_step_name in unique_names:
                        count += 1
                        modified_step_name = _generate_modified_step(step_name, count)
                    modified_name_step = (modified_step_name,) + step[1:]
                    unique_names.add(modified_step_name) 
                    return modified_name_step, unique_names
                
                
            new_step_list, unique_names = [], set()
            # For each dict {'step': {params}} in provided list
            for step in step_list:
                # Generate numbering for the step if step use multiple times in pipeline. E.g custom-1 and custom-2 if there are two 'custom' steps.
                new_step, unique_names = _generate_unique_name(step, unique_names)
                new_step_list.append(new_step)
            return new_step_list
    
        def _add_steps_to_pipeline(steps_to_add=None, add_to_pipeline=None):
            """
            Add custom processing steps to a scikit-learn pipeline.
            Will wrap the passed transformer if its name does not begin by '*'. If it does, the transformer will be used in its native passed form.

            Parameters:
            - steps_to_add (list): List of tuples defining steps to be added.
            - add_to_pipeline, optional (Pipeline): Existing scikit-learn pipeline. If None, an empty one is created. 

            Returns:
            - add_to_pipeline (Pipeline): Extended pipeline.
            """
            # Check if steps_to_add is valid
            if (steps_to_add is None) or (not steps_to_add):
                raise ValueError(f'No steps_to_add ({steps_to_add}) to add to the pipeline.')

            # Create a new pipeline if add_to_pipeline is None
            add_to_pipeline = Pipeline(steps=[]) if add_to_pipeline is None else add_to_pipeline
                

            # Iterate through the steps to be added
            for name, transformer_or_functransformer, parameters, *transformer_wrapper_parameters in steps_to_add:
                transformer_wrapper_parameters = transformer_wrapper_parameters[0] if transformer_wrapper_parameters else {}

                # Handle special case where name starts with '*' so that the transformer is added in its native form
                if name[0] == '*':
                    name = name[1:]
                    transformer = transformer_or_functransformer(**parameters)
                    
                # Handle case where transformer_or_functransformer is a function_transformer
                elif isinstance(transformer_or_functransformer, types.FunctionType):
                    transformer = FunctionTransformerTransformer(transformer_or_functransformer,
                                                                 **parameters
                                                                 )
                # Handle case where transformer_or_functransformer is a scikit-learn transformer that will be wrapped in DynamicFeatureTransformer
                elif issubclass(transformer_or_functransformer, BaseEstimator):
                    transformer = DynamicFeatureTransformer(transformer_or_functransformer,
                                                            **transformer_wrapper_parameters,
                                                            **parameters
                                                            )
                # Append the step to the pipeline
                add_to_pipeline.steps.append((name, transformer))
            return add_to_pipeline
    
        def _actualize_attributes(created_pipeline):
            """
            Update class attributes with the newly created or extended pipeline.

            Parameters:
            - created_pipeline (Pipeline): Pipeline to be added to class attributes.

            Returns:
            - None
            """
            self.data_pipelines_.append((name, created_pipeline))
            return
    
        # If pipeline if provided, assign it as attribute            
        if pipeline is not None and pipeline:
            _actualize_attributes(pipeline)
        
        # Else if steps (list of dicts {'step': {params}}) is provided
        elif steps is not None and steps:
            # Create the pipeline based on steps and assign it as attribute
            preprocessing_steps = _add_counter_step_names(steps)
            new_pipeline = _add_steps_to_pipeline(steps_to_add=preprocessing_steps)
            _actualize_attributes(new_pipeline)   
            
        return new_pipeline
    

    def remove_dropped_index_in_y(self, X, y):
        '''
        Method that removes rows in y (target variable) corresponding to the indices not present in X (features).
        It basically enables to run sampling preprocessing inside our pipeline. (e.g outlier removal)
        '''
        y = y.loc[X.index]
        return y 
        
        
    def apply_data_pipeline(self, pipeline_tuple, X_fit_tr=None, X_transform=None):
        def run_preprocessing(pipeline, to_fit_transform=None, to_transform=None):
            if pipeline is None: raise ValueError('Args[0] "pipeline" must not be None.')

            if to_fit_transform is not None:
                return pipeline.fit_transform(to_fit_transform)
            elif to_transform is not None:
                return pipeline.transform(to_transform)

        pipeline_name, pipeline = pipeline_tuple

        if all(df is not None for df in [X_fit_tr, X_transform]):
            X_fit_tr = run_preprocessing(pipeline, to_fit_transform=X_fit_tr)
            X_transform = run_preprocessing(pipeline, to_transform=X_transform)  
            return X_fit_tr, X_transform        
                    
        elif X_fit_tr is not None:
            return run_preprocessing(pipeline, to_fit_transform=X_fit_tr)
                               
        elif X_transform is not None:
            return run_preprocessing(pipeline, to_transform=X_transform)




class MLpPlaygroundPreprocessing():
    def __init__(self, X_copy, y_copy):
        """
        Initializes the MLpPlaygroundPreprocessing instance. Aimed at helping trying different sampling and preprocessing strategies.
        Parameters (both are mandatory):
        - initial_X_copy: Initial X df copy.
        - y_copy: Copy of the target variable.
        """
        if X_copy is not None and y_copy is not None:
            self.X_copy = X_copy
            self.y_copy = y_copy
            # Check if target is continoous or not. If yes convert it as numerical column
            try:
                self.y_copy = pd.to_numeric(self.y_copy)
            except ValueError:
                pass
        else :
            raise ValueError('Need to provide an initial X and y dataframe to use the playground.')

        self.mlp_preprocessing_instance = MLpPreprocessing()
        
        self.count_pipelines_pg = 0
        self.list_pipeline_pg = []
        
        self.all_X_y_pg = []
        self.X_pg = None
        self.y_pg = None
        
    def get_transformed_X_and_y(self):
        return self.X_pg, self.y_pg
    
    def playground_transform(self, name, steps={}, pipeline=None):
        """
        Perform playground transformations (applying sampling or preprocessing) on X df (self.X_pg).
        Parameters (need to provide type + steps or pipeline):
        - type: Type of transformation ('sampling' or 'preprocessing').
        - steps: Dictionary of transformation steps.
        - pipeline: Custom transformation pipeline.
        Returns:
        - DataFrame: Transformed feature dataset.
        """
        def _update_attributes(pipeline_to_add):
            """
            Update attributes to track the transformation steps performed.
            Build self.list_pipeline_pg (encompasses all playground_transform call pipelines).
            Parameters:
            - pipeline_to_add: Pipeline that has just been applied
            """
            self.count_pipelines_pg += 1
            self.list_pipeline_pg.append((f'{name}-{self.count_pipelines_pg}', pipeline_to_add))
            self.all_X_y_pg.append((self.X_pg, self.y_pg)) 
            return
        
        # If X_pg is None, use the initial X df
        if self.X_pg is None:
            self.X_pg = self.X_copy.copy()
            self.y_pg = self.y_copy.copy()
            
        #  Create a 'type' pipeline based on provided 'steps' or 'pipeline'
        if steps:
            my_pipeline = self.mlp_preprocessing_instance.create_data_pipeline(name=name, steps=steps)
        elif pipeline is not None and pipeline:
            my_pipeline = self.mlp_preprocessing_instance.create_data_pipeline(name=name, pipeline=pipeline)
        else:
            raise ValueError('Please provide the parameters "name" and either "steps" or "pipeline".')
            
        # Apply the transformation
        self.X_pg = my_pipeline.fit_transform(self.X_pg)
        self.y_pg = self.y_pg.loc[self.X_pg.index]
        _update_attributes(my_pipeline)
        return self.X_pg, self.y_pg
    
    
    def delete_previous_transformations(self, n_steps:int):
        """
        Delete previous transformation steps performed.
        Parameters:
        - n_steps: Number of steps to delete (from the last transformation step).
        """
        if n_steps :
            # Remove n_steps to count and select remaining in list_pipeline_pg, all_X_y_pg
            self.count_pipelines_pg -= n_steps
            self.list_pipeline_pg = self.list_pipeline_pg[:-n_steps]
            self.all_X_y_pg = self.all_X_y_pg[:-n_steps]
            # Get the last X_pg and y_pg
            if self.all_X_y_pg:
                self.X_pg, self.y_pg = self.all_X_y_pg[-1] 
            else:
                self.X_pg, self.y_pg = self.X_copy, self.y_copy
        else:
            raise ValueError('Need n_steps parameter (int()) to delete previous steps.')    
        return self.X_pg, self.y_pg
    
