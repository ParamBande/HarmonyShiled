import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class ModelTrainer:
    """
    Handles classical ML training and persistence for Plan A.
    """
    def __init__(self, model_type='rf', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._build_model()
        
    def _build_model(self):
        models = {
            'lr': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'svm': SVC(probability=True, random_state=self.random_state),
            'rf': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'dt': DecisionTreeClassifier(random_state=self.random_state),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'gnb': GaussianNB(),
            'gbdt': HistGradientBoostingClassifier(random_state=self.random_state)
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        clf = models[self.model_type]
        
        # Always use a StandardScaler in the pipeline for robust classical ML
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
        
        return pipeline
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        if hasattr(self.model.named_steps['classifier'], "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model does not support probability prediction.")
            
    def get_feature_importances(self, feature_names=None):
        clf = self.model.named_steps['classifier']
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            if feature_names is not None:
                return dict(zip(feature_names, importances))
            return importances
        else:
            return None
            
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        
    @classmethod
    def load(cls, filepath):
        instance = cls()
        instance.model = joblib.load(filepath)
        return instance
