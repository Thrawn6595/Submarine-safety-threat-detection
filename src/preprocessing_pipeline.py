"""
Preprocessing Pipeline for ML Models
Handles algorithm-specific scaling requirements
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict


class ModelPreprocessor:
    """Preprocessing pipeline that applies appropriate scaling per algorithm"""
    
    SCALER_MAP = {
        'logistic_regression': 'standard',
        'svm': 'standard',
        'random_forest': 'none',
        'knn': 'standard',
        'naive_bayes': 'none',
        'catboost': 'none',
        'lda': 'standard',
        'qda': 'standard'
    }
    
    def __init__(self):
        self.scalers = {}
    
    def get_scaler_for_algorithm(self, algorithm: str):
        """Return appropriate scaler for algorithm"""
        scaler_type = self.SCALER_MAP.get(algorithm.lower(), 'standard')
        
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            return None
    
    def fit_transform(self, X_train: np.ndarray, algorithm: str) -> np.ndarray:
        """Fit scaler on training data and transform"""
        scaler = self.get_scaler_for_algorithm(algorithm)
        
        if scaler is None:
            self.scalers[algorithm] = None
            return X_train
        
        X_scaled = scaler.fit_transform(X_train)
        self.scalers[algorithm] = scaler
        return X_scaled
    
    def transform(self, X_test: np.ndarray, algorithm: str) -> np.ndarray:
        """Transform test data using fitted scaler"""
        scaler = self.scalers.get(algorithm)
        
        if scaler is None:
            return X_test
        
        return scaler.transform(X_test)
    
    def get_preprocessing_info(self) -> Dict:
        """Return preprocessing information for all algorithms"""
        info = {}
        for algo, scaler_type in self.SCALER_MAP.items():
            info[algo] = {
                'scaler_type': scaler_type,
                'fitted': algo in self.scalers
            }
        return info


def prepare_data_for_algorithm(X_train: np.ndarray, X_test: np.ndarray, 
                               algorithm: str, preprocessor: ModelPreprocessor = None) -> Tuple:
    """Prepare data with appropriate scaling for specific algorithm"""
    if preprocessor is None:
        preprocessor = ModelPreprocessor()
    
    X_train_scaled = preprocessor.fit_transform(X_train, algorithm)
    X_test_scaled = preprocessor.transform(X_test, algorithm)
    
    return X_train_scaled, X_test_scaled, preprocessor
