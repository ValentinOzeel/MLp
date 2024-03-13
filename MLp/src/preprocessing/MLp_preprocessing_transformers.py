import itertools

from MLp.conf.config_functions import get_config
mlp_config = get_config()
RANDOM_STATE = mlp_config['MLP_CONFIG']['RANDOM_STATE']

from MLp.src.preprocessing.transformer_decorators import get_num_cat_features_decorator, select_columns_decorator

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

from MLp.src.secondary_modules.import_libraries import import_cpu_gpu_sklearn, import_cpu_gpu_pandas
pd = import_cpu_gpu_pandas()

SimpleImputer, KNNImputer = import_cpu_gpu_sklearn('impute', ['SimpleImputer', 'KNNImputer'])
OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer = import_cpu_gpu_sklearn('preprocessing', ['OneHotEncoder', 
                                                                                                                                                             'LabelEncoder', 
                                                                                                                                                             'StandardScaler', 
                                                                                                                                                             'MinMaxScaler', 
                                                                                                                                                             'RobustScaler', 
                                                                                                                                                             'PowerTransformer', 
                                                                                                                                                             'QuantileTransformer'])
KMeans, AgglomerativeClustering, DBSCAN = import_cpu_gpu_sklearn('cluster', ['KMeans', 'AgglomerativeClustering', 'DBSCAN'])
PCA = import_cpu_gpu_sklearn('decomposition', 'PCA')


class DropColsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):     
        self.columns = columns        
        self.selected_cols = []

    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.selected_cols:
            X = X.drop(self.selected_cols, axis=1)
        return X



class SplitExpandTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, feature_splits={}):
        self.columns = columns
        self.feature_splits = feature_splits
        
        self.selected_cols = []
        self.new_features = []

    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def _recognize_numeric_columns(expanded_columns):
            ### Method that recognizes numeric columns and convert their type
            def _convert_to_numeric(column):
                #### Method that converts a column to numeric if possible
                try:
                    return pd.to_numeric(column)
                except ValueError:
                    return column

            # Loop over newly expanded columns
            for col in expanded_columns.columns:
                # Convert expoanded column to numeric if possible (If expanded col is string of numbers)
                expanded_columns[col] = _convert_to_numeric(expanded_columns[col])
            return expanded_columns
        
        for feature, dict_value in self.feature_splits.items():
            if isinstance(dict_value, str):
                split_char, new_col_names = dict_value, None 
            elif isinstance(dict_value, list):
                split_char, new_col_names = dict_value[0], dict_value[1] 
            else:
                raise ValueError('"feature_split" parameter of SplitExpandTransformer should be a Dict with entries:\n\t\t - key: a string "ColumnName"\n\t\t - value: a string "splitting_character" (e.g "-") OR a list ["splitting_character", [new_column_names]}')

            # Calculate the number of non-NaN values in the column
            count_non_nan_values = X[feature].count()
            # Count of values containing the splitting character, excluding NaN values
            count_with_split_char = (
                X[feature]
                .apply(lambda x: split_char in str(x) if pd.notna(x) else False)
                .sum()
            )
            # % of values that contain the splitting char, excluding NaN
            percentage_with_split = (count_with_split_char / count_non_nan_values) * 100
            # If at least 97.5% of values contains the split char
            if percentage_with_split > 97.5:

                # Split and expand the columns
                new_columns = X[feature].str.split(split_char, expand=True)
                # Rename expanded columns
                if new_col_names is not None and len(new_col_names) == len(new_columns.columns):
                    new_columns.columns = new_col_names
                else:
                    new_columns.columns = [f"{feature}_{i+1}" for i in range(new_columns.shape[1])]
                # Update new_features attribute
                self.new_features = new_columns.columns
                # Recognize potential numerical expanded columns
                new_columns = _recognize_numeric_columns(new_columns)
                # Drop the inital column that got split and expanded (cat_col)
                # Concat newly created columns to the original dataframe
                X = pd.concat([X.drop(feature, axis=1), new_columns], axis=1)
        return X
    
    







class SimpleImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, strategy=None):
        self.columns = columns        
        self.strategy = strategy
        
        self.selected_cols = []
        self.imputer = None

    
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        # Create SimpleImputer with selected strategy
        self.imputer = SimpleImputer(strategy=self.strategy)
        # Fit SimpleImputer on the training data
        self.imputer.fit(X[self.selected_cols])
        return self

    def transform(self, X):
        # Transform selected columns with the SimpleImputer and reassign correct columns names and indexing lost during transformation
        imputed_X = pd.DataFrame(
            self.imputer.transform(X[self.selected_cols]),
            columns=self.selected_cols,
            index=X.index
        )
        # Concatenate the transformed numerical columns with the initial columns
        X = pd.concat([X.drop(columns=self.selected_cols), imputed_X], axis=1)
        return X


class KNNImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, n_neighbors=5):
        
        self.columns = columns        
        self.n_neighbors = n_neighbors
        
        self.selected_cols = []
        self.knn_imputer = None

    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
        # Fit the KNNImputer on the specified columns
        self.knn_imputer.fit(X[self.selected_cols])
        return self
    
    def transform(self, X):
        # Transform categorical columns with the SimpleImputer
        imputed_X = pd.DataFrame(
            self.knn_imputer.transform(X[self.selected_cols]),
            columns=self.selected_cols,
            index=X.index
        )
        # Concatenate the transformed categorical columns with the initial numerical columns
        X = pd.concat(
            [X.drop(columns=self.selected_cols), imputed_X], axis=1
        )
        return X

    
    
    

class OHEncodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None,
                       max_cardinality=15,
                       handle_unknown="ignore",
                       sparse_output=False):

        self.columns = columns        
        self.max_cardinality = max_cardinality
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        
        self.selected_cols = [] 
        self.oh_encoder = None
        self.oh_encoded_cols = []
        self.new_features = []

    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        # Select features to OH encode (those with less than max_cardinality unique values)
        OH_enc = [col for col in self.selected_cols if X[col].nunique() < self.max_cardinality]
        if OH_enc:
            # Create and fit the OneHot encoder
            self.oh_encoder = OneHotEncoder(
                handle_unknown=self.handle_unknown, 
                sparse_output=self.sparse_output,
                drop='first'
            )
            self.oh_encoder.fit(X[OH_enc])
            # Get the names of the new encoded features
            self.oh_encoded_cols = OH_enc
        return self

    def transform(self, X):
        if self.oh_encoded_cols:
            # Transform the selected features using the pre-fitted OneHot encoder
            OH_cols = pd.DataFrame(self.oh_encoder.transform(X[self.oh_encoded_cols]),
                                   columns=self.oh_encoder.get_feature_names_out(self.oh_encoded_cols),
                                   index=X.index)
            # Update new_features attribute
            self.new_features = OH_cols.columns
            # Drop the original columns that were OneHot encoded
            # Concatenate X with the OH encoded columns
            return pd.concat(
                [X.drop(columns=self.oh_encoded_cols), OH_cols], axis=1
            )
        




class LabelEncodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, min_cardinality=15):
    
        self.columns = columns        
        self.min_cardinality = min_cardinality
        
        self.selected_cols = [] 
        self.label_encoders = {}
        self.label_encoded_cols = []

    
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        initial_cols = [
            col for col in self.selected_cols if X[col].nunique() >= self.min_cardinality
        ]
        if initial_cols:
            for col in initial_cols:
                label_encoder = LabelEncoder()
                label_encoder.fit(X[col])
                self.label_encoders[col] = label_encoder
        self.label_encoded_cols = self.label_encoders.keys()
        return self

    def transform(self, X):        
        label_encoded_cols = pd.DataFrame()
        if self.label_encoded_cols:
            for col, label_encoder in self.label_encoders.items():
                if col in X.columns:
                    # Label encode each value in X[col] while Handling unknown categories in X_test that were not seen suring training
                    lab_col = X[col].apply(
                        lambda x: label_encoder.transform([x])[0]
                        if x in label_encoder.classes_
                        else -1
                    )
                    lab_col.columns = col
                    lab_col.index = X.index
                    # Store the label-encoded column for later concatenation
                    label_encoded_cols[col] = lab_col
        # Drop the original columns that were label-encoded
        # Concatenate the label-encoded columns with the original DataFrame (excluding the labelized cols)
        return pd.concat(
            [X.drop(label_encoded_cols.columns, axis=1), label_encoded_cols], axis=1
        )





class BinningTransformer(BaseEstimator, TransformerMixin):
    
    '''
    custom_binning should be in the form : 
    {'labels':[],
    'bins':[]}
    '''
    
    def __init__(self, columns=None,
                       equal_frequency_binning=False,
                       equal_width_binning=False,
                       n_groups=None,
                       custom_binning={},
                       bins_as_numerical=False,
                       drop=False):
        
        self.columns = columns        
        self.equal_frequency_binning = equal_frequency_binning
        self.equal_width_binning = equal_width_binning
        self.n_groups = n_groups
        self.custom_binning = custom_binning
        self.bins_as_numerical = bins_as_numerical
        self.drop = drop
        
        self.selected_cols = [] 
        self.col_bins = {}
        self.col_labels = {}
          
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        for col in self.selected_cols:
            # Calculate the bin edges for splitting into groups to perform percentile-based discretization of a continuous variable : BASED ON DATA DISTRIBUTION
            # Divide the data into self.n_groups equal-width bins based on percentiles. Each bin represents approximately self.n_groups% of the data.
            #eg this type of bins : [ 0., 22., 33., 79.] : three groups of 0.33, 0.33 0.33 %

            if self.equal_frequency_binning and not self.equal_width_binning:
                # Calculate the bin edges for each specified column
                _, bins = pd.qcut(X[col], q=self.n_groups, retbins=True, labels=False)
                self.col_bins[col] = bins
                self.col_labels[col] = range(self.n_groups)
                
            # Calculate the bin edges for splitting into groups to perform minmax-based discretization of a continuous variable : BASED ON DATA MIN AND MAX VALUES
            #4.g this type of bins : [ 0., 10., 20., 30.] if min = 0 and max = 30,  : three groups of 0.45, 0.35 0.2 %
            elif self.equal_width_binning and not self.equal_frequency_binning:
                _, bins = pd.cut(X[col], bins=self.n_groups, labels=False, include_lowest=True, retbins=True)
                self.col_bins[col] = bins
                self.col_labels[col] = range(self.n_groups)
            
            elif self.custom_binning:                    
                self.col_bins[col] = self.custom_binning['bins']
                self.col_labels[col] = self.custom_binning['labels']
        return self
    
    def transform(self, X):
        for col, bins in self.col_bins.items():
            # Add a new column with group labels
            X[f'{col}_bins'] = pd.cut(X[col], bins=bins, labels=self.col_labels[col], include_lowest=True)
            
            if self.bins_as_numerical:
                # Use LabelEncoder to convert categorical labels to float64
                X[f'{col}_bins'] = LabelEncoder().fit_transform(X[f'{col}_bins']).astype('float64')
                
        if self.drop: X = X.drop(self.col_bins.keys(), axis=1)
        return X
    
    
    
class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, preprocess_=False):
        
        self.columns=columns        
        self.preprocess_=preprocess_

        self.selected_cols = [] 
        self.minmax_scaler = None        
    
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        # Initialize MinMaxScaler
        self.minmax_scaler = MinMaxScaler()
        # Fit the scaler on the training data
        self.minmax_scaler.fit(X[self.selected_cols])
        return self
    
    def transform(self, X):
        # Transform the numerical columns using the fitted scaler
        minmaxed = pd.DataFrame(
            self.minmax_scaler.transform(X[self.selected_cols]), 
            columns=self.selected_cols,
            index=X.index
        )
        # Drop initial numerical columns
        # Concatenate initial categorical columns with selected_cols (MinMaxed) numerical columns
        return pd.concat([X.drop(self.selected_cols, axis=1), minmaxed], axis=1)
    

    
    
class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, preprocess_=False):
 
        self.columns=columns
        self.preprocess_=preprocess_
                       
        self.selected_cols = [] 
        self.standard_scaler = None
    
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        # Create and fit StandardScaler
        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(X[self.selected_cols])
        return self
    
    # Transform numerical columns using the parameters learned during fit
    def transform(self, X):
        # Transform numerical columns using the stored StandardScaler
        standardscaled = pd.DataFrame(
            self.standard_scaler.transform(X[self.selected_cols]),
            columns=self.selected_cols,
            index=X.index,
        )
        # Drop initial numerical columns
        # Concatenate initial categorical columns with selected_cols (StandardScaled) numerical columns
        return pd.concat([X.drop(self.selected_cols, axis=1), standardscaled], axis=1)
        

    

#This scaler is robust to outliers as it scales data based on the interquartile range (IQR)
class RobustScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, preprocess_=False):
        
        self.columns = columns
        self.preprocess_ = preprocess_

        self.selected_cols = [] 
        self.robust_scaler = None
        self.transformed = []
    
    @get_num_cat_features_decorator
    @select_columns_decorator  
    def fit(self, X, y=None):
        # Initialize RobustScaler
        self.robust_scaler = RobustScaler()
        # Fit the scaler on the training data
        self.robust_scaler.fit(X[self.selected_cols])
        return self
    
    def transform(self, X):
        # Transform the numerical columns using the fitted scaler
        robust_scaled = pd.DataFrame(
            self.robust_scaler.transform(X[self.selected_cols]), 
            columns=self.selected_cols,
            index=X.index
        )
        # Drop initial numerical columns
        # Concatenate initial categorical columns with selected_cols (Robust Scaled) numerical columns
        return pd.concat([X.drop(self.selected_cols, axis=1), robust_scaled], axis=1)
    
    
    
    
    


class PowerBoxCoxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, preprocess_=False):
        
        self.columns = columns
        self.preprocess_ = preprocess_

        self.selected_cols = [] 
        self.power_transformer = None
        self.transformed = []
        
    @get_num_cat_features_decorator
    @select_columns_decorator  
    def fit(self, X, y=None):
        # Initialize PowerTransformer
        self.power_transformer = PowerTransformer(method='box-cox')
        # Fit the transformer on the training data
        self.power_transformer.fit(X[self.selected_cols])
        return self
    
    def transform(self, X):
        # Transform the numerical columns using the fitted transformer
        power_transformed = pd.DataFrame(
            self.power_transformer.transform(X[self.selected_cols]), 
            columns=self.selected_cols,
            index=X.index
        )
        # Drop initial numerical columns
        # Concatenate initial categorical columns with selected_cols (Power Transformed) numerical columns
        return pd.concat([X.drop(self.selected_cols, axis=1), power_transformed], axis=1)
        



class PowerYeoJohnsonScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, preprocess_=False):
        
        self.columns = columns        
        self.preprocess_ = preprocess_

        self.selected_cols = [] 
        self.power_transformer = None
        
    @get_num_cat_features_decorator
    @select_columns_decorator  
    def fit(self, X, y=None):
        # Initialize PowerTransformer
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        # Fit the transformer on the training data
        self.power_transformer.fit(X[self.selected_cols])
        return self
    
    def transform(self, X):
        # Transform the numerical columns using the fitted transformer
        power_transformed = pd.DataFrame(
            self.power_transformer.transform(X[self.selected_cols]), 
            columns=self.selected_cols,
            index=X.index
        )
        # Drop initial numerical columns
        # Concatenate initial categorical columns with selected_cols (Power Transformed) numerical columns
        return pd.concat([X.drop(self.selected_cols, axis=1), power_transformed], axis=1)
        
    



class QuantileScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, preprocess_=False):
        
        self.columns = columns
        self.preprocess_ = preprocess_

        self.selected_cols = [] 
        self.quantile_transformer = None
                
    @get_num_cat_features_decorator
    @select_columns_decorator 
    def fit(self, X, y=None):
        # Initialize QuantileTransformer
        self.quantile_transformer = QuantileTransformer()
        # Fit the transformer on the training data
        self.quantile_transformer.fit(X[self.selected_cols])
        return self
    
    def transform(self, X):
        # Transform the numerical columns using the fitted transformer
        quantile_transformed = pd.DataFrame(
            self.quantile_transformer.transform(X[self.selected_cols]), 
            columns=self.selected_cols,
            index=X.index
        )
        # Drop initial numerical columns
        # Concatenate initial categorical columns with selected_cols (Quantile Transformed) numerical columns
        return pd.concat([X.drop(self.selected_cols, axis=1), quantile_transformed], axis=1)
    
    

class LogScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, preprocess_=False):
        
        self.columns = columns
        self.preprocess_ = preprocess_

        self.selected_cols = [] 

    @get_num_cat_features_decorator
    @select_columns_decorator  
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Apply log transformation to numerical columns
        for col in self.selected_cols:
            log_col = np.log1p(X[col])     
            # Drop initial numerical columns
            # Concatenate initial categorical columns with selected_cols (StandardScaled) numerical columns
            X = pd.concat([X.drop(col, axis=1), log_col], axis=1)
        return X


    
    
    



class OperationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None,
                       single_col_operation='', # ['power', 'square_root', 'cude_root', 'exponential', 'cos', 'sin', 'tan', 'log']
                       power_operator=int(),
                       multiple_cols_method='', # ['addition', 'substration', 'range', 'product', 'ratio',  'mean', 'median', 'variance', 'std']
                       aggregation= True,
                       feature_name=None,
                       combinations= int(),
                       drop=False
    ):
        self.columns = columns
        self.drop = drop
        # Single col operation
        self.single_col_operation = single_col_operation
        # To apply power
        self.power_operator = power_operator 
        # Multiple cols methods
        self.multiple_cols_method = multiple_cols_method
        # Apply operation on columns aggregation
        self.aggregation = aggregation
        # Apply operation on all combinations of n
        self.combinations = combinations
        self.feature_name = feature_name
        self.available_single_col_operations = ['power', 'square_root', 'cude_root', 'exponential', 'cos', 'sin', 'tan', 'log']
        self.available_multiple_cols_methods = ['sum', 'substration', 'range', 'product', 'ratio',  'mean', 'median', 'variance', 'std_dev']
        
        self.selected_cols = [] 
        
    @get_num_cat_features_decorator
    @select_columns_decorator  
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.single_col_operation and self.single_col_operation in self.available_single_col_operations:
            X = self._carry_out_single_col_operation(X, self.selected_cols)
        
        elif self.aggregation:
            X = self._multiple_cols_methods(X, self.selected_cols, 'aggr', name=self.feature_name)
            
        elif self.combinations:
            for combination in list(itertools.combinations(self.selected_cols, self.combinations)):
                X = self._multiple_cols_methods(X, list(combination), 'comb')
            
        if self.drop: X = X.drop(self.selected_cols, axis=1)
            
        return X
            

    def _multiple_cols_methods(self, X, cols, prefix, name=None):

        if self.multiple_cols_method not in self.available_multiple_cols_methods:
            raise ValueError(f'Unsupported multiple_cols_method: {self.multiple_cols_method}')

        if name is None:
            name_cols = '_'.join(cols)
            name = f'{prefix}_{name_cols}'

        if self.multiple_cols_method == 'mean':
            X[f'mean_{name}'] = X[cols].mean(axis=1)
        elif self.multiple_cols_method == 'median':
            X[f'median_{name}'] = X[cols].median(axis=1)
        elif self.multiple_cols_method == 'variance':
            X[f'variance_{name}'] = X[cols].var(axis=1)
        elif self.multiple_cols_method == 'std_dev':
            X[f'std_dev_{name}'] = X[cols].std(axis=1)

        # Addition, Subtraction, Range, Product, Ratio
        elif self.multiple_cols_method == 'sum':
            X[f'sum_{name}'] = X[cols].sum(axis=1)
        elif self.multiple_cols_method == 'substration':
            X[f'substration_{name}'] = X[cols].iloc[:, 0] - X[cols].iloc[:, 1:].sum(axis=1)
            reverse = cols[::-1]
            name_cols_r = '_'.join(reverse)
            name = f'{prefix}_{name_cols_r}'
            X[f'substration_{name}'] = X[reverse].iloc[:, 0] - X[reverse].iloc[:, 1:].sum(axis=1)
        elif self.multiple_cols_method == 'range':
            X[f'range_{name}'] = X[cols].max(axis=1) - X[cols].min(axis=1)
        elif self.multiple_cols_method == 'product':      
            X[f'product_{name}'] = X[cols].prod(axis=1)

        if self.multiple_cols_method == 'ratio' :
            if len(cols) == 2:
                # .replace(0, 1) avoid division by zero in the 'Ratio' calculation
                if prefix == 'comb':
                    X[f'ratio_{name}'] = X[cols[0]] / X[cols[1]].replace(0, 1)
                    reverse = cols[::-1]
                    name_cols_r = '_'.join(reverse)
                    name = f'{prefix}_{name_cols_r}'
                    X[f'ratio_{name}'] = X[reverse[0]] / X[reverse[1]].replace(0, 1)
                else:
                    X[f'ratio_{name}'] = X[cols[0]] / X[cols[1]].replace(0, 1)
            else:
                raise ValueError('The ratio is computed between two values. Please provide two columns or make use of the combination parameter.')

        return X


    def _carry_out_single_col_operation(self, X, cols):
        if self.single_col_operation not in self.available_single_col_operations:
            raise ValueError(f'Unsupported single_col_operation: {self.single_col_operation}')

        for col in cols:  
            if self.single_col_operation == 'power':    
                if self.power_operator:
                    X[f'{col}_power'] = np.power(X[col], self.power_operator)
                else:
                    raise ValueError('Need to provide an operator if using power. Make sure to provide the parameter power_operator.')
            elif self.single_col_operation == 'sqrt':
                X[f'{col}_sqrt'] = np.sqrt(X[col])
            elif self.single_col_operation == 'cbrt':
                X[f'{col}_cbrt'] = np.cbrt(X[col])
            elif self.single_col_operation == 'exp':
                X[f'{col}_exp'] = np.exp(X[col])
            elif self.single_col_operation == 'cos':
                X[f'{col}_cos'] = np.cos(X[col])
            elif self.single_col_operation == 'sin':
                X[f'{col}_sin'] = np.sin(X[col])
            elif self.single_col_operation == 'tan':
                X[f'{col}_tan'] = np.tan(X[col])
            elif self.single_col_operation == 'log':
                X[f'{col}_log'] = np.log1p(X[col])

        return X
    
    
    

    
class RoundTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, decimals=2):
        
        self.columns = columns
        self.decimals = decimals
        
        self.selected_cols = [] 

    @get_num_cat_features_decorator
    @select_columns_decorator  
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Round specified columns
        for col in self.selected_cols:
            round_col = X[col].round(decimals=self.decimals)
            X = pd.concat([X.drop(col, axis=1), round_col], axis=1)
        return X
    












def _get_preprocess_transform (X, preprocess_transformer):
    # Get the full preprocessed df with preprocessed columns and non preprocessed columns
    X_preprocessed = preprocess_transformer.fit_transform(X)
    # Reassign indexing lost during transofrmation
    X_preprocessed.index = X.index
    return X_preprocessed, preprocess_transformer

class KMeansClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None,
                       preprocess_transformer=None,
                       preprocess_params={},
                       drop=False,
                       n_clusters=None,
                       random_state=RANDOM_STATE
    ):
             
        self.columns = columns
        self.preprocess_transformer = preprocess_transformer() if preprocess_transformer is not None else None
        self.preprocess_params = preprocess_params if preprocess_params else {}
        if self.preprocess_transformer is not None and self.preprocess_params:
            self.preprocess_transformer.set_params(**self.preprocess_params)
        self.drop = drop
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.selected_cols = [] 
        self.kmeans_model = None  # Store the fitted KMeans model

    
    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):
        if self.preprocess_transformer:
            # Get process transformer, preprocessed df and preprocessed cols
            X_preprocessed, self.preprocess_transformer = _get_preprocess_transform(X[self.selected_cols], self.preprocess_transformer)
            # Fit the model on the preprocessed columns
            self.kmeans_model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(X_preprocessed)
            
        else:
            self.kmeans_model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(X[self.selected_cols])
        return self
     
    def transform(self, X):
        if self.preprocess_transformer:
            X_preprocessed = pd.DataFrame(self.preprocess_transformer.transform(X[self.selected_cols]), index=X.index)
            new_feature = pd.DataFrame(self.kmeans_model.predict(X_preprocessed), columns=['kmeans'], index=X.index).astype('float64')
        else:
            new_feature = pd.DataFrame(self.kmeans_model.predict(X[self.selected_cols]), columns=['kmeans'], index=X.index).astype('float64')

        if self.drop:
            X = X.drop(self.selected_cols, axis=1)
            
        return pd.concat([X, new_feature], axis=1)







class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns=None,
        preprocess_transformer=None,
        preprocess_params={},
        drop=False,
        n_components=None,
        component_names=None,
        random_state=RANDOM_STATE
    ):
        
        self.columns = columns
        self.preprocess_transformer = preprocess_transformer() if preprocess_transformer is not None else None
        self.preprocess_params = preprocess_params if preprocess_params else {}
        if self.preprocess_transformer is not None and self.preprocess_params:
            self.preprocess_transformer.set_params(**self.preprocess_params)
        self.drop = drop
        self.n_components = n_components
        self.component_names = component_names
        self.random_state = random_state
        
        self.selected_cols = []
        self.pca_model = None  # Store the fitted pca model
        self.preprocess_transformer = None # Store the transformer used for preprocessing

    @get_num_cat_features_decorator
    @select_columns_decorator
    def fit(self, X, y=None):

        if self.preprocess_transformer:
            # Get process transformer, preprocessed df and preprocessed cols
            X_preprocessed, self.preprocess_transformer = _get_preprocess_transform(X[self.selected_cols], self.preprocess_transformer)
            # Fit the model only with the preprocessed columns
            self.pca_model = PCA(n_components=self.n_components, random_state=self.random_state).fit(X_preprocessed)
  
        else:
            self.pca_model = PCA(n_components=self.n_components, random_state=self.random_state).fit(X[self.selected_cols])
        return self
     
    def transform(self, X):

        if self.preprocess_transformer:
            X_preprocessed = self.preprocess_transformer.transform(X[self.selected_cols])
            new_feature = self.pca_model.transform(X_preprocessed)

        else:
            new_feature = self.pca_model.transform(X[self.selected_cols])

        # Convert pca_result back to DataFrame
        column_names = self.component_names if self.component_names is not None else [f"pca_comp_{i+1}" for i in range(new_feature.shape[1])]
        pca_result = pd.DataFrame(
            data=new_feature,
            columns=column_names,
            index=X.index,
        )

        if self.drop:
            X = X.drop(self.selected_cols, axis=1)
            
        X = pd.concat([X, pca_result], axis=1)

        return X
















class AgglomerativeClusteringTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self, columns=None,
                       preprocess_transformer=None,
                       preprocess_params={},
                        drop=False, 
                        n_clusters=None):
        
        
        self.columns = columns
        self.preprocess_transformer = preprocess_transformer() if preprocess_transformer is not None else None
        self.preprocess_params = preprocess_params if preprocess_params else {}
        if self.preprocess_transformer is not None and self.preprocess_params:
            self.preprocess_transformer.set_params(**self.preprocess_params)
        self.drop = drop
        self.n_clusters = n_clusters
        
        self.selected_cols = []
        

    # THERE IS NO PREDICT NOR TRANSFORM METHOD FOR AGG, ONLY fit_predict
    @get_num_cat_features_decorator
    @select_columns_decorator    
    def fit(self, X, y=None):

        if self.preprocess_transformer:
            # Get process transformer, preprocessed df and preprocessed cols
            _, self.preprocess_transformer = _get_preprocess_transform(X[self.selected_cols], self.preprocess_transformer)
                    
        return self
     
    def transform(self, X):
        model = AgglomerativeClustering(n_clusters=self.n_clusters)
        if self.preprocess_transformer:
            X_preprocessed = self.preprocess_transformer.transform(X[self.selected_cols])
            new_feature = model.fit_predict(X_preprocessed)

        else:
            new_feature = model.fit_predict(X[self.selected_cols])

        # Convert pca_result back to DataFrame
        agg_result = pd.DataFrame(
            data=new_feature,
            columns=["agglomerative_cluster"],
            index=X.index,
        )

        if self.drop:
            X = X.drop(self.agg_cluster_cols, axis=1)
            
        return pd.concat([X, agg_result], axis=1)














class DBSCANClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns=None,
        preprocess_transformer=None,
        preprocess_params={},
        drop=False,
        min_samples=None,
        eps=None,
    ):
        
        self.columns = columns
        self.preprocess_transformer = preprocess_transformer() if preprocess_transformer is not None else None
        self.preprocess_params = preprocess_params if preprocess_params else {}
        if self.preprocess_transformer is not None and self.preprocess_params:
            self.preprocess_transformer.set_params(**self.preprocess_params)
        self.drop = drop
        self.min_samples = min_samples
        self.eps = eps
        
        self.selected_cols = []
            
        
    # THERE IS NO PREDICT NOR TRANSFORM METHOD FOR DBscan, ONLY fit_predict
    @get_num_cat_features_decorator
    @select_columns_decorator 
    def fit(self, X, y=None):
        if self.preprocess_transformer:
            # Get process transformer, preprocessed df and preprocessed cols
            _, self.preprocess_transformer = _get_preprocess_transform(X[self.selected_cols], self.preprocess_transformer)
              
        return self
     
    def transform(self, X):
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        if self.preprocess_transformer:
            X_preprocessed = pd.DataFrame(self.preprocess_transformer.transform(X[self.selected_cols]), index=X.index)
            new_feature = pd.DataFrame(model.fit_predict(X_preprocessed), columns=['dbscan_cluster'], index=X.index)
        else:
            new_feature = pd.DataFrame(model.fit_predict(X[self.selected_cols]), columns=['dbscan_cluster'], index=X.index)

        if self.drop:
            X = X.drop(self.selected_cols, axis=1)
            
        return pd.concat([X, new_feature], axis=1)

