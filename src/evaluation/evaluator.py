import json
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, classification_report
)

class ModelEvaluator:
    """
    Evaluates Plan A models and generates metrics.
    """
    @staticmethod
    def evaluate(y_true, y_pred, y_prob=None, output_dir=None, prefix="eval"):
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Use macro average to treat both classes equally
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        if y_prob is not None:
            try:
                # Assuming binary classification for ROC AUC, extracting positive class prob
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except Exception:
                pass
                
        metrics['report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"{prefix}_metrics.json")
            with open(json_path, 'w') as f:
                json.dump(metrics, f, indent=4)
                
            ModelEvaluator.plot_confusion_matrix(cm, os.path.join(output_dir, f"{prefix}_cm.png"))
            
        return metrics
        
    @staticmethod
    def plot_confusion_matrix(cm, filepath):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(filepath)
            plt.close()
        except ImportError:
            pass # Skip if matplotlib/seaborn are not available
