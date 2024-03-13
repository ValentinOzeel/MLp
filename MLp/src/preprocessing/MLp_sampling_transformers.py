from MLp.conf.config_functions import get_config
mlp_config = get_config()
RANDOM_STATE = mlp_config['MLP_CONFIG']['RANDOM_STATE']

from MLp.src.preprocessing.transformer_decorators import get_num_cat_features_decorator, select_columns_decorator

from MLp.src.secondary_modules.import_libraries import import_cpu_gpu_sklearn
BaseEstimator, TransformerMixin = import_cpu_gpu_sklearn('base', ['BaseEstimator', 'TransformerMixin'])
IsolationForest = import_cpu_gpu_sklearn('ensemble', 'IsolationForest')

import numpy as np





class DropNaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns        
        self.selected_cols = []

    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.selected_cols:
            X = X.dropna(subset=self.selected_cols, axis=0)
        return X


############# OUTLIERS ###############################

class IQROutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, IQR_multiplier=1.5):

        self.columns = columns        
        self.IQR_multiplier = IQR_multiplier

        self.selected_cols = [] 
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        for col in self.selected_cols:
            data_column = X[col]
            # Calculate the first and third quartiles (Q1 and Q3)
            Q1 = np.percentile(data_column, 25)
            Q3 = np.percentile(data_column, 75)
            # Calculate the IQR (Interquartile Range)
            IQR = Q3 - Q1
            # Define the lower and upper bounds for outliers
            lower_bound = Q1 - self.IQR_multiplier * IQR
            upper_bound = Q3 + self.IQR_multiplier * IQR
            # Store the lower and upper bounds for later use in transform
            self.lower_bounds_[col] = lower_bound
            self.upper_bounds_[col] = upper_bound
        return self

    def transform(self, X):
        # Initialize an empty mask that represents the union of all the individual masks of each column
        total_outliers_mask = np.zeros(len(X), dtype=bool)

        for col in self.selected_cols:
            data_column = X[col]
            # Identify outliers for the current column based on the stored bounds
            outliers = (data_column < self.lower_bounds_[col]) | (data_column > self.upper_bounds_[col])
            # Update the total outlier mask
            total_outliers_mask |= outliers
              
        # Return the DataFrame without outliers
        return X[~total_outliers_mask]
        

class WinsorizerOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None,
                       low_percentile_threshold=5, 
                       high_percentile_threshold=95):

        self.columns = columns
        self.low_percentile_threshold = low_percentile_threshold
        self.high_percentile_threshold = high_percentile_threshold

        self.selected_cols = [] 
        self.percentile_values = {}

    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        for col in self.selected_cols:
            # Calculate and store the specified percentile values during fit
            self.percentile_values[col] = {
                'lower_bound': np.percentile(X[col], self.low_percentile_threshold),
                'upper_bound': np.percentile(X[col], self.high_percentile_threshold),
            }
        return self

    def transform(self, X):
        for col in self.selected_cols:
            # Retrieve the stored percentile values for the current column
            lower_bound = self.percentile_values[col]['lower_bound']
            upper_bound = self.percentile_values[col]['upper_bound']
            # Winsorize values beyond the specified percentiles
            X[col] = np.clip(X[col], lower_bound, upper_bound)
        return X



class TruncationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None,
                       low_threshold=None, 
                       high_threshold=None):

        self.columns = columns        
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self.selected_cols = [] 
    
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        return self

    # Method that replace value higher than high_threshold by high_threshold and lower than low_threshold by low_threshold
    def transform(self, X):
        for col in self.selected_cols:
            # Truncate values beyond the specified thresholds
            if self.low_threshold is not None:
                X[col] = np.where(
                    X[col] < self.low_threshold, self.low_threshold, X[col]
                )
            if self.high_threshold is not None:
                X[col] = np.where(
                    X[col] > self.high_threshold, self.high_threshold, X[col]
                )
        return X



class ZScoreOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, threshold=3):
        
        self.columns = columns        
        self.threshold = threshold

        self.selected_cols = [] 
        self.mean_std_dict = {}  # Dictionary that stores mean and std for each column
    
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        for col in self.selected_cols:
            # Calculate mean and standard deviation for each column
            mean_value = X[col].mean()
            std_value = X[col].std()
            # Store the mean and std in the dictionary
            self.mean_std_dict[col] = {'mean': mean_value, 'std': std_value}
        return self
    
    # Method that remove outliers based on zscore
    def transform(self, X):
        for col in self.mean_std_dict:
            # Retrieve mean and std from the stored values
            mean_value = self.mean_std_dict[col]['mean']
            std_value = self.mean_std_dict[col]['std']
            # Calculate Z-scores using mean and std from the training data
            z_scores = np.abs((X[col] - mean_value) / std_value)
            # Identify outliers based on Z-scores
            outliers = z_scores > self.threshold
            # Remove outliers
            X = X[~outliers]
        return X
    

class IsolationForestOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None,
                       contamination=0.05,
                       random_state=RANDOM_STATE):
        
        self.columns = columns        
        self.contamination = contamination
        self.random_state = random_state
        
        self.selected_cols = [] 
        self.isol_forest = None
        
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        self.isol_forest = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        self.isol_forest.fit(X[self.selected_cols])
        return self

    # Return a binary mask indicating whether each sample is an inlier (1) or an outlier (-1)
    def transform(self, X):
        # Identify outliers using the Isolation Forest
        outlier_mask = self.isol_forest.predict(X[self.selected_cols]) == -1
        # Remove outliers from X
        X = X[~outlier_mask]
        return X