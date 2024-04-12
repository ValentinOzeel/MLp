import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from MLp.conf.config_functions import get_config
mlp_config = get_config()
RANDOM_STATE = mlp_config['MLP_CONFIG']['RANDOM_STATE']
from MLp.src.preprocessing.transformer_decorators import get_num_cat_features_decorator, select_columns_decorator
from MLp.src.secondary_modules.import_libraries import import_cpu_gpu_sklearn

IsolationForest = import_cpu_gpu_sklearn('ensemble', 'IsolationForest')

'''
    -- AVAILABLE DATA SAMPLING TRANSFORMER CLASSES (Follow the sklearn nomenclature / Users can also write their own custom transformers and use them in MLp) --
    
    DROP NaN VALUES
    -----------
    - DropNaTransformer
    
    OUTLIER REMOVAL
    -----------
    - IQROutliersTransformer
    - WinsorizerOutliersTransformer
    - TruncationTransformer
    - ZScoreOutliersTransformer
    - IsolationForestOutliersTransformer
'''




class DropNaTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop rows with missing values (NaNs) in selected columns.

    Parameters:
    -----------
    columns : list or None, optional (default=None)
        List of column names to consider. If None, all columns will be considered.

    """

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
    """
    Custom transformer to remove outliers from numerical columns using IQR (Interquartile Range).
    
    Parameters:
    -----------
    columns : list or None, default=None
        Columns to apply outlier removal. If None, applies to all numerical columns.
    IQR_multiplier : float, default=1.5
        Multiplier to adjust the range for detecting outliers.
    """
    
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
            Q1 = np.nanpercentile(data_column, 25)
            Q3 = np.nanpercentile(data_column, 75)
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
    """
    A transformer to winsorize outliers in numerical columns.

    Parameters:
    -----------
    columns : list, optional (default=None)
        List of column names to apply winsorization. If None, all numerical columns are selected.
    low_percentile_threshold : int or float, optional (default=5)
        Lower percentile threshold for winsorization.
    high_percentile_threshold : int or float, optional (default=95)
        Upper percentile threshold for winsorization.
    """
    
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
    """
    A transformer class for truncating numerical values in specified columns
    beyond given thresholds.

    Parameters:
    -----------
    low_threshold : float
        Lower threshold for truncation. Values below this threshold will
        be replaced by `low_threshold`. If None, truncation will not be
        applied for values below any threshold.

    high_threshold : float
        Upper threshold for truncation. Values above this threshold will
        be replaced by `high_threshold`. If None, truncation will not be
        applied for values above any threshold.
        
    columns : list or None, optional (default=None)
        List of column names to apply truncation. If None, all numerical
        columns will be considered for truncation.
    """
    
    def __init__(self, low_threshold:float, 
                       high_threshold:float,
                       columns=None,
                       ):

        self.columns = columns        
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self.selected_cols = [] 
    
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        return self

    # Transform the input data by truncating values in selected columns. Replace value higher than high_threshold by high_threshold and lower than low_threshold by low_threshold
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
    """
    Custom transformer to remove outliers from numerical columns using Z-scores.
    
    Parameters:
    ----------
    columns : list, optional
        List of column names to consider. If None, all numerical columns will be considered.
    threshold : float, optional (default=3)
        Threshold value for identifying outliers. 
    """
    
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
    """
    Custom transformer to remove outliers using Isolation Forest algorithm.

    Parameters:
    -----------
    columns : list, default=None
        List of column names to consider for outlier detection. If None, all columns will be considered.
        
    contamination : float, default=0.05
        The amount of contamination in the data set, i.e., the proportion of outliers in the data set.
        
    random_state : int, default=RANDOM_STATE
        Random seed for reproducibility.
    """
    
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

    def transform(self, X):
        # With isolation forest, generate a binary mask indicating whether each sample is an inlier (1) or an outlier (-1)
        outlier_mask = self.isol_forest.predict(X[self.selected_cols]) == -1
        # Remove outliers from X
        X = X[~outlier_mask]
        return X