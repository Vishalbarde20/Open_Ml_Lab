from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class ModelRegistry:
    """Registry of available ML models with their hyperparameters"""
    
    @staticmethod
    def get_available_models():
        return {
            "Random Forest": RandomForestClassifier,
            "Logistic Regression": LogisticRegression,
            "SVM": SVC,
            "Decision Tree": DecisionTreeClassifier,
            "K-Nearest Neighbors": KNeighborsClassifier,
            "Gradient Boosting": GradientBoostingClassifier
        }
    
    @staticmethod
    def get_hyperparameters(model_name):
        """Return hyperparameter configuration for each model"""
        params = {
            "Random Forest": {
                "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100},
                "max_depth": {"type": "slider", "min": 1, "max": 50, "default": 10},
                "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2},
                "criterion": {"type": "selectbox", "options": ["gini", "entropy"], "default": "gini"}
            },
            "Logistic Regression": {
                "C": {"type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.01},
                "max_iter": {"type": "slider", "min": 100, "max": 1000, "default": 100},
                "solver": {"type": "selectbox", "options": ["lbfgs", "liblinear", "saga"], "default": "lbfgs"}
            },
            "SVM": {
                "C": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
                "kernel": {"type": "selectbox", "options": ["rbf", "linear", "poly"], "default": "rbf"},
                "gamma": {"type": "selectbox", "options": ["scale", "auto"], "default": "scale"}
            },
            "Decision Tree": {
                "max_depth": {"type": "slider", "min": 1, "max": 50, "default": 5},
                "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2},
                "criterion": {"type": "selectbox", "options": ["gini", "entropy"], "default": "gini"}
            },
            "K-Nearest Neighbors": {
                "n_neighbors": {"type": "slider", "min": 1, "max": 50, "default": 5},
                "weights": {"type": "selectbox", "options": ["uniform", "distance"], "default": "uniform"},
                "algorithm": {"type": "selectbox", "options": ["auto", "ball_tree", "kd_tree"], "default": "auto"}
            },
            "Gradient Boosting": {
                "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100},
                "learning_rate": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
                "max_depth": {"type": "slider", "min": 1, "max": 10, "default": 3}
            }
        }
        return params.get(model_name, {})
