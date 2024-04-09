from typing import List

# Method decorator: Identify numerical and categorical features in the dataframe to preprocess
def get_num_cat_features_decorator(func):
    """
    A decorator function to automatically retrieve numerical and categorical features 
    from a DataFrame before executing a method.

    Returns:
        callable: The decorated method.
    """
    def wrapper(self, *args, **kwargs):
        # Get num and cat features
        self.num_features = list(args[0].select_dtypes(include=["int", "float"]).columns)
        self.cat_features = list(args[0].select_dtypes(include=["object", "category"]).columns) 
        # Run the method
        return func(self, *args, **kwargs)
    return wrapper    
        
            
def select_columns_decorator(func):
    """
    A decorator function for selecting columns before applying preprocessing transformers.
    The self.columns of the decorated transformer should be:
    - None or 'all' to select all columns
    - a list of columns to select 
    - a list with first element = '~' to select all columns except these listed afterward.
    - str 'categoricals' or 'numericals' to select all categorical or numerical columns.

    Returns:
        callable: The decorated function.
    """
    def wrapper(self, *args, **kwargs):
        columns = self.columns if hasattr(self, 'columns') else self._columns
        
        # If it's a preprocess for another transformer (e.g scaling before applying cluster or pca transformer): consider all columns
        if hasattr(self, 'preprocess_') and self.preprocess_:
            self.selected_cols = columns
            return func(self, *args, **kwargs)
            
        # Get all existing column names
        all_cols = [element for sublist in [self.num_features, self.cat_features] for element in sublist]   

        # If columns is None: consider all columns
        if columns is None:
            self.selected_cols =  all_cols
            
        # Else, if columns is List
        elif isinstance(columns, List) and columns:
            # Check if there is '~' as first element
            if columns[0] == '~':
                # Get all columns EXCEPT these specified in columns
                self.selected_cols = [col for col in all_cols if col not in columns.pop(0)]
            else:   
                # Get all columns specified in columns
                self.selected_cols =  columns

        # Elif columns is str
        elif isinstance(columns, str) and columns:
            if columns.lower() == 'all':
                self.selected_cols =  all_cols
            elif columns.lower() == 'numericals':
                self.selected_cols =  [col for col in self.num_features]
            elif columns.lower() == 'categoricals':
                self.selected_cols =  [col for col in self.cat_features]
                              
        if not self.selected_cols:
            raise ValueError(f'\nPreprocessing cannot be carried out: no column to apply {func.__name__} on.\n')

        return func(self, *args, **kwargs)
    return wrapper


