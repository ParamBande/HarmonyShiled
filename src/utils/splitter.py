import numpy as np
from sklearn.model_selection import GroupShuffleSplit

class DataSplitter:
    """
    Handles leakage-safe dataset splitting at the track/group level.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def split_data(self, X, y, groups, test_size=0.15, val_size=0.15):
        """
        Splits data into train, val, and test sets safely by group.
        
        Args:
            X: Features
            y: Labels
            groups: Array of group IDs (e.g., track IDs)
            test_size: Proportion of dataset to include in the test split
            val_size: Proportion of dataset to include in the validation split
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test
        """
        X = np.array(X)
        y = np.array(y)
        groups = np.array(groups)
        
        # First split: separate out the test set
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
        train_val_idx, test_idx = next(gss_test.split(X, y, groups))
        
        X_train_val, y_train_val, groups_train_val = X[train_val_idx], y[train_val_idx], groups[train_val_idx]
        X_test, y_test, g_test = X[test_idx], y[test_idx], groups[test_idx]
        
        # Adjust val_size relative to the train_val portion
        relative_val_size = val_size / (1.0 - test_size)
        
        # Second split: separate out the val set from train_val
        gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=self.random_state)
        train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
        
        X_train, y_train, g_train = X_train_val[train_idx], y_train_val[train_idx], groups_train_val[train_idx]
        X_val, y_val, g_val = X_train_val[val_idx], y_train_val[val_idx], groups_train_val[val_idx]
        
        return X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test
