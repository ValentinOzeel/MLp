from sklearn.pipeline import Pipeline
import pandas as pd


# Method decorator: raise ValueError if the necessary kwargs are not assigned 
def validate_initialize_data_intputs(func):
    def wrapper(self, *args, **kwargs):

        condition_to_run = [['data', 'target'], ['X', 'y']]
        for condition in condition_to_run:
            sliced_kwargs = {key: kwargs.get(key, None) for key in condition}
            if all([value is not None for key,value in sliced_kwargs.items()]):
                # Run the method
                return func(self, *args, **kwargs)          
        raise ValueError(f'\nMethod: {func.__name__}\nAll kwargs of one of the following lists should be assigned to run the method:\n{condition_to_run}') 
    return wrapper  

# Method decorator: raise ValueError if the necessary kwargs are not assigned 
def validate_at_least_one_kwarg_not_none(func):
    def wrapper(self, *args, **kwargs):
        if any([value is not None for key,value in kwargs.items()]):
            return func(self, *args, **kwargs)
        else:
            raise ValueError(f'\nMethod: {func.__name__}\nAt least on kwargs should be assigned to run the method.\n')
    return wrapper  

# Method decorator: raise ValueError if the necessary kwargs are not assigned 
def validate__run_sampling_preprocess(func):
    def wrapper(self, *args, **kwargs):
        '''
        self.sampling_pipeline_ must be already created.
        Run the decorated method if one of these conditions is true:
        - X_train, y_train: is not None   AND   X_test,y_test: is None
        - X_train, X_test, y_train, y_test: is not None
        - X_to_predict: is not None
        '''
        if not isinstance(args[0], Pipeline) or not args[0]:
            raise ValueError(f'\nMethod: {func.__name__}\nargs[0] should be an instance of sklearn Pipeline that samples (and potentially preprocesses) data.\n')
        
        kw_X_train, kw_X_test, kw_y_train, kw_y_test, kw_X_to_predict = (
            kwargs.get(key, None) for key in ['X_train', 'X_test', 'y_train', 'y_test', 'X_to_predict']
        )
        
        if ((kw_X_train is not None) and (kw_y_train is not None))  and  ((kw_X_test is None) and (kw_y_test is None)) \
            or (all(value is not None for value in [kw_X_train, kw_X_test, kw_y_train, kw_y_test])) \
            or (kw_X_to_predict is not None):
            return func(self, *args, **kwargs)
        else:
            raise ValueError(f'\nMethod: {func.__name__}\nAt least one of these kwargs combinations should be assigned:\n\t- X_train, y_train: is not None   AND   X_test,y_test: is None\n\t- X_train, X_test, y_train, y_test: is not None\n\t- X_to_predict: is not None')
    return wrapper  

# Method decorator: raise ValueError if the necessary kwargs are not assigned 
def validate__run_preprocessing(func):
    def wrapper(self, *args, **kwargs):
        '''
        args[0] should be an instance of sklearn Pipeline that preprocesses data
        Run the decorated method if one of these conditions is true:
        - X_train, X_test: is not None
        - X_train is not None and X_test is None
        - X_train is None and X_test is not None
        - X_to_predict is not None
        '''
        if not isinstance(args[0], Pipeline) or not args[0]:
            raise ValueError(f'\nMethod: {func.__name__}\nargs[0] should be an instance of sklearn Pipeline that preprocesses data.\n')
        
        kw_X_train, kw_X_test, kw_X_to_predict = (
            kwargs.get(key, None) for key in ['X_train', 'X_test', 'X_to_predict']
        )
        
        if (kw_X_train is not None) and (kw_X_test is not None) \
            or (kw_X_train is not None) and (kw_X_test is None) \
            or (kw_X_train is None) and (kw_X_test is not None) \
            or  kw_X_to_predict is not None:
                return func(self, *args, **kwargs)
        else:
            raise ValueError(f'\nMethod: {func.__name__}\nAt least one of these kwargs combinations should be assigned:\n\t- X_train, X_test: is not None\n\t- X_train is not None and X_test is None\n\t- X_train is None and X_test is not None\n\t- X_to_predict is not None\n')
    return wrapper   

# Method decorator: raise ValueError if there is no preprocessing_pipeline_ or model_ defined
def validate_defined_data_pipeline(func):
    def wrapper(self, *args, **kwargs):
        if not self.created_pipelines_:
            raise ValueError (f'\nMethod: {func.__name__}\nPlease first create at least one data pipeline with the "create_data_pipeline" method of the MLpBuilder class first.')

        # Run the method
        return func(self, *args, **kwargs)
    return wrapper    

# Method decorator: raise ValueError if the necessary kwargs are not assigned 
def validate__split_data(func):
    def wrapper(self, *args, **kwargs):        
        X_cond = isinstance(args[0], pd.DataFrame)
        y_cond = isinstance(args[1], pd.DataFrame) or isinstance(args[1], pd.Series)
        test_size_cond = isinstance(args[2], float) and (0<args[2]<1)

        if all([cond for cond in [X_cond, y_cond, test_size_cond]]):
            return func(self, *args, **kwargs)
        else:
            raise ValueError(f'\nMethod: {func.__name__}\nargs[0]: pd.DataFrame, args[1]: pd.DataFrame or pd.Series, args[2]: 0<float<1, kwargs["shuffle"]: True or False\n')
    return wrapper  







def validate_apply_created_data_pipelines(func):
    def wrapper(self, *args, **kwargs):
        '''
        Run the decorated method if these conditions are true:
        - self.created_pipelines_ is defined and not empty
        - 'X_train', 'X_test', 'y_train', 'y_test' parameters are passed
          OR 'X', 'y'
          OR 'X_to_predict'
        - parameters are Dataframe or Series or ndarray
        - Dataframe or Series are not empty

        '''
        
        if not self.created_pipelines_:
            raise ValueError(f'\nMethod: {func.__name__}\nPlease first create at least one data pipeline with the "create_data_pipeline" method of the MLpBuilder class first.\n')
        
        if not kwargs:
            raise ValueError(f'\nMethod: {func.__name__}\nPlease pass parameters.\n')




        kw_X_train, kw_X_test, kw_y_train, kw_y_test, kw_X, kw_y, kw_X_to_predict = (
            kwargs.get(key, None) for key in ['X_train', 'X_test', 'y_train', 'y_test', 'X', 'y', 'X_to_predict']
        )
        
        
        to_check = None
        if all([False for val in [kw_X_train, kw_X_test, kw_y_train, kw_y_test] if val is None]):
            to_check = [kw_X_train, kw_X_test, kw_y_train, kw_y_test]
        elif (kw_X is not None) and (kw_y is not None):
            to_check = [kw_X, kw_y]
        elif kw_X_to_predict is not None:
            to_check = [kw_X_to_predict]
            
        if to_check is None:
            raise ValueError(f'\nMethod: {func.__name__}\nCombination of passed kwargs should be one of the following :\n- X_train, X_test, y_train, y_test\n- X, y\n- X_to_predict')
        
        
        import numpy as np
        for data in to_check:
            if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
                raise TypeError(f'\nMethod: {func.__name__}\nKwarg {data} must be either DataFrame, Series or ndarray.')
            if data.empty:
                raise ValueError(f'\nMethod: {func.__name__}\nKwarg {data} cannot be empty.')

        # If all checks pass, execute the original function
        return func(self, *args, **kwargs)

    return wrapper












def validate__run_cross_validation(func):
    def wrapper(self, *args, **kwargs):
        '''
        Run the decorated method if these conditions are true:
        - self.created_pipelines_ is defined and not empty
        - 4 args are defined
        - args[0] and [1], X and y, are not empty DF/Series
        - args[2], kf, should be a kf objet
        - args[3], scoring, should be a string representing scoring method
        - kwargs n_jobs should be an int number  
        '''
        
        if not self.created_pipelines_:
            raise ValueError(f'\nMethod: {func.__name__}\nPlease first create at least one data pipeline with the "create_data_pipeline" method of the MLpBuilder class first.\n')
        

        # Check if all arguments are passed
        if len(args) != 4:
            raise ValueError(f'\nMethod: {func.__name__}\nAll 4 args should be assigned in the following order "X", "y", "kf", "scoring".\n')

        # Check if arguments are DataFrames or Series
        for count, arg in enumerate([args[0], args[1]]):
            if not isinstance(arg, (pd.DataFrame, pd.Series)):
                raise TypeError(f'\nMethod: {func.__name__}\Arg{count} must be either DataFrame or Series.')
            # Check if DataFrame or Series is not empty
            if arg.empty:
                raise ValueError(f'\nMethod: {func.__name__}\nArg{count} cannot be an empty DataFrame/Series.')

        if not args[2]:
            raise ValueError(f'\nMethod: {func.__name__}\nArg2 should be a defined kfold objet.')
        
        if not isinstance(args[3], str):
            raise ValueError(f'\nMethod: {func.__name__}\nArg3 should be a string representing the scoring method.')
        
        if kwargs.get("n_jobs", None) is not None:
            if not isinstance(kwargs.get("n_jobs", None), int):
                raise ValueError(f'\nMethod: {func.__name__}\nKwarg "n_jobs" should be an int number.')  
            
            
        # If all checks pass, execute the original function
        return func(self, *args, **kwargs)

    return wrapper



# Method decorator: raise ValueError if the necessary kwargs are not assigned 
def validate_optuna(func):
    def wrapper(self, *args, **kwargs):        
        if self.pre_split is None:
            raise ValueError(f'\nMethod: {func.__name__}\nargs[0]: pd.DataFrame, args[1]: pd.DataFrame or pd.Series, args[2]: 0<float<1, kwargs["shuffle"]: True or False\n')
    return wrapper  



# Method decorator: raise ValueError if the necessary kwargs are not assigned 
def validate_grid_search(func):
    def wrapper(self, *args, **kwargs):        
        if self.pre_split is None:
            raise ValueError(f'\nMethod: {func.__name__}\nargs[0]: pd.DataFrame, args[1]: pd.DataFrame or pd.Series, args[2]: 0<float<1, kwargs["shuffle"]: True or False\n')
    return wrapper  

