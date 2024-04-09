from MLp.conf.config_functions import get_config
mlp_config = get_config()
RANDOM_STATE = mlp_config['MLP_CONFIG']['RANDOM_STATE']
from MLp.src.secondary_modules.import_libraries import import_cpu_gpu_sklearn, import_cpu_gpu_pandas

pd = import_cpu_gpu_pandas()
# Scoring metrics for Classifier  
accuracy_score, log_loss, precision_score, recall_score, f1_score = import_cpu_gpu_sklearn('metrics', ['accuracy_score', 
                                                                                                          'log_loss', 
                                                                                                          'precision_score', 
                                                                                                          'recall_score', 
                                                                                                          'f1_score'])
# Scoring metrics for Regressor
mean_squared_error, mean_absolute_error, r2_score = import_cpu_gpu_sklearn('metrics', ['mean_squared_error', 
                                                                                          'mean_absolute_error', 
                                                                                          'r2_score' ])
# Feature importance
permutation_importance = import_cpu_gpu_sklearn('inspection', 'permutation_importance')
# Feature selection
mutual_info_regression, mutual_info_classif = import_cpu_gpu_sklearn('feature_selection', ['mutual_info_regression', 'mutual_info_classif'])
# Calibration
CalibratedClassifierCV = import_cpu_gpu_sklearn('calibration', 'CalibratedClassifierCV')
# Import clone
clone = import_cpu_gpu_sklearn('base', 'clone')

import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
         
       
class MLpModel():
    def __init__(self, *args, **kwargs):    
        # XGB model
        self.model_ = None
        self.model_name_ = None

        # Create a mapper for scoring metrics to their corresponding functions
        self.scoring_mapper = {
            'accuracy': accuracy_score,
            'log_loss': log_loss,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            
            'mean_squared_error': mean_squared_error,
            'mean_absolute_error': mean_absolute_error,
            'r2': r2_score,
        } 
        # Scoring metrics that are minimilizing, so need to multiple them by -1
        self.minimilizing_scoring_metrics = ['log loss', 
                                             'mean squared error', 'mse',
                                             'mean absolute error', 'mae']
        

    # Method that creates the model
    def define_model(self, use_model, model_name, calibration='', calib_cv=''):
        """
        Method that defines and potentially calibrates self.model_.
        Parameters:
        - use_model: Model class to use.
        - model_name: Name of the model
        - calibration: Calibration method ('platt_scaling' or 'isotonic_regression') if calibration is needed.
        - calib_cv: Number of cross-validation folds for calibration.
        """
        def _apply_calibration(calibration=None, cv=5):
            """
            Applies calibration to self.model_.
            Parameters:
            - calibration: Calibration method ('platt_scaling' or 'isotonic_regression').
            - cv: Number of cross-validation folds for calibration.
            """
            if calibration.lower() in ['platt_scaling', 'platt', 'platt scaling']:
                self.model_ = CalibratedClassifierCV(self.model_, method='sigmoid', cv=cv)
            elif calibration.lower() in ['isotonic_regression', 'isotonic', 'isotonic regression']:
                self.model_ = CalibratedClassifierCV(self.model_, method='isotonic', cv=cv)

        if (use_model is not None and use_model) and (model_name is not None and model_name):
            self.model_ = use_model
            self.model_name_ = model_name
        else:
            raise ValueError('No model is provided. Please provide a model through the "use_model" parameter.')
            
        if calibration:
            _apply_calibration(calibration=calibration, cv=calib_cv)
            
        return self.model_


    def evaluate_model_get_score(self, scoring_metric, y_test, predictions):
        """ 
        Evaluate model's test score based on the specified scoring metric.
        Parameters:
        - scoring_metric: The scoring metric to use.
        - y_test: The true labels.
        - predictions: The predicted labels.
        Returns:
        - score: The calculated score.
        
        Scoring : 
                Classifier (
                           accuracy, accuracy_score : The ratio of correctly predicted observation to the total observations. SHOULD BE MAXIMIZED
                           log loss, log_loss : It measures the performance of a classification model where the prediction is a probability value between 0 and 1. SHOULD BE MINIMIZED (multiply score by -1)
                           precision, precision_score : The ratio of correctly predicted positive observations to the total predicted positives. Relevant for imbalanced datasets. SHOULD BE MAXIMIZED
                           recall (Sensitivity or True Positive Rate), recall_score : The ratio of correctly predicted positive observations to all the observations in the actual class. SHOULD BE MAXIMIZED
                           f1, f1_score : The weighted average of Precision and Recall. SHOULD BE MAXIMIZED
                         )
               Regressor (
                          Mean Squared Error (MSE), mean_squared_error: Measures the average squared difference between the predicted and true values. SHOULD BE MINIMIZED (multiply score by -1)
                          Mean Absolute Error (MAE), mean_absolute_error: Measures the average absolute difference between the predicted and true values. SHOULD BE MINIMIZED (multiply score by -1)
                          R-squared (R2), r2_score : Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. SHOULD BE MAXIMIZED
                         )
        """
        scoring_metric = scoring_metric.lower()
        score = 0

        # Find the key in the scoring_mapper that matches the given scoring_metric
        scoring_function = self.scoring_mapper.get(scoring_metric)
        # Compute the test score
        if scoring_function is not None:
            score = scoring_function(y_test, predictions)
        else:
            raise ValueError(f'No or incorrect "scoring_metric" provided. Available:{self.scoring_mapper.keys()}\n')

        # Multiply by -1 if scoring_metric is a minimizing metric
        return -1 * score if scoring_metric in self.minimilizing_scoring_metrics else score
    


    def check_feature_importance(self, method='', scoring='', X=None, y=None):
        """
        Check feature importance using various methods and visualize the results.
        Parameters:
        - method: The method used for feature importance ('built_in', 'permutation_importance', 'drop_column_importance',
                  'shap', 'mutual_info_regression', 'mutual_info_classif').
        - scoring: Scoring metric for evaluating model performance.
        - X: Feature df.
        - y: Target variable.
        Returns:
        - None
        """
        def _calculate_drop_column_importance(X, y, scoring):
            """
            Calculate feature importance by dropping each column one at a time and measuring the impact on the model's
            performance.
            Parameters:
            - X: Feature df.
            - y: Target variable.
            - scoring: Scoring metric for evaluating model performance.
            Returns:
            - feature_importance: List of feature importance scores.
            """
            baseline_score = self.evaluate_model_get_score(scoring, y, self.model_.predict(X), verbose=False)
            feature_importance = []
            for col in X.columns:
                X_temp = X.drop(columns=col)
                model_temp = clone(self.model_)
                model_temp.fit(X_temp, y)
                score_temp = self.evaluate_model_get_score(scoring, y, model_temp.predict(X_temp), verbose=False)
                feature_importance.append(baseline_score - score_temp)
            return feature_importance

        def _calculate_shap_importance(X):
            """
            Calculate feature importance using SHAP values.
            Parameters:
            - X: Feature df.
            Returns:
            - feature_importance: List of feature importance scores.
            """
            explainer = shap.Explainer(self.model_)
            shap_values = explainer.shap_values(X)
            return np.abs(shap_values).mean(axis=0)

        def _calculate_mutual_info_importance(X, y, method_lower):
            """
            Calculate feature importance using mutual information.
            Parameters:
            - X: Feature matrix.
            - y: Target variable.
            - method_lower: Lowercased method name ('mutual_info_regression' or 'mutual_info_classif').
            Returns:
            - feature_importance: List of feature importance scores.
            """
            if method_lower == 'mutual_info_regression':
                return mutual_info_regression(X, y)
            elif method_lower == 'mutual_info_classif':
                return mutual_info_regression(X, y)
            
        def _visualize_importance(feature_names, importance, sort_ascending=False):
            # Create a DataFrame for better visualization
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=sort_ascending)
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Feature Importance')
            plt.show()
            return
        
        if not method:
            raise ValueError('Cannot check feature importance as no "method" parameter is provided.')        
        
        method_lower = method.lower()

        if method_lower == 'built_in':
            feature_importance = self.model_.feature_importances_
        elif method_lower == 'permutation_importance':
            result = permutation_importance(self.model_, X, y, n_repeats=30, random_state=RANDOM_STATE)
            feature_importance = result.importances_mean
        elif method_lower == 'drop_column_importance':
            feature_importance = _calculate_drop_column_importance(X, y, scoring)
        elif method_lower == 'shap':
            feature_importance = _calculate_shap_importance(X)
        elif method_lower in ['mutual_info_regression', 'mutual_info_classif']:
            feature_importance = _calculate_mutual_info_importance(X, y, method_lower)
        else:
            raise ValueError(f"Unsupported method: {method}")

        _visualize_importance(X.columns, feature_importance, sort_ascending=False)
        
        return
        

            



