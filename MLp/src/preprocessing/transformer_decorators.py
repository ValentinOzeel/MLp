from typing import List

# Method decorator: Identify numerical and categorical features in the dataframe to preprocess
def get_num_cat_features_decorator(func):
    def wrapper(self, *args, **kwargs):
        # Get num and cat features
        self.num_features = list(args[0].select_dtypes(include=["int", "float"]).columns)
        self.cat_features = list(args[0].select_dtypes(include=["object", "category"]).columns) 
        # Run the method
        return func(self, *args, **kwargs)

    return wrapper    
        
        
        
def select_columns_decorator(func):
    def wrapper(self, *args, **kwargs):
        
        columns = self.columns if hasattr(self, 'columns') else self._columns
        
        
        if hasattr(self, 'preprocess_') and self.preprocess_:
            self.selected_cols = columns
            return func(self, *args, **kwargs)
            
        # Get all existing column names
        all_cols = [element for sublist in [self.num_features, self.cat_features] for element in sublist]   

        # If columns is None
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
            if columns == 'all':
                self.selected_cols =  all_cols
            elif columns == 'numericals':
                self.selected_cols =  [col for col in self.num_features]
            elif columns == 'categoricals':
                self.selected_cols =  [col for col in self.cat_features]
                           
        
        if not self.selected_cols:
            raise ValueError(f'\nPreprocessing cannot be carried out: no column to apply {func.__name__} on.\n')

        return func(self, *args, **kwargs)
    return wrapper


