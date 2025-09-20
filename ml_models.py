"""
Machine Learning Models for Train Arrival Time and Stop Duration Prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from config import DATA_CONFIG, MODEL_CONFIG

class TrainArrivalPredictor:
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.best_models = {}
        self.feature_importance = {}
        
    def initialize_models(self):
        """Initialize all ML models for both tasks"""
        
        # Models for Travel Time Prediction
        self.models['travel_time'] = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        # Models for Stop Duration Prediction
        self.models['stop_duration'] = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
    
    def train_models(self, X_train, y_train, X_test, y_test, task_name):
        """Train all models for a specific task"""
        print(f"\nTraining models for {task_name} prediction...")
        
        task_models = self.models[task_name]
        scores = {}
        
        for model_name, model in task_models.items():
            print(f"Training {model_name}...")
            
            try:
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()
                
                scores[model_name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_mae': cv_mae,
                    'cv_std': cv_std
                }
                
                print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
                scores[model_name] = None
        
        self.model_scores[task_name] = scores
        return scores
    
    def find_best_models(self):
        """Find the best performing models for each task"""
        for task_name in ['travel_time', 'stop_duration']:
            if task_name in self.model_scores:
                scores = self.model_scores[task_name]
                valid_scores = {k: v for k, v in scores.items() if v is not None}
                
                if valid_scores:
                    # Find best model based on MAE (lower is better)
                    best_model = min(valid_scores.items(), key=lambda x: x[1]['mae'])
                    self.best_models[task_name] = {
                        'name': best_model[0],
                        'model': self.models[task_name][best_model[0]],
                        'scores': best_model[1]
                    }
                    print(f"Best model for {task_name}: {best_model[0]} (MAE: {best_model[1]['mae']:.2f})")
    
    def hyperparameter_tuning(self, X_train, y_train, task_name, model_name):
        """Perform hyperparameter tuning for the best model"""
        print(f"\nPerforming hyperparameter tuning for {model_name} on {task_name}...")
        
        if model_name == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_name == 'lightgbm':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return self.models[task_name][model_name]
        
        model = self.models[task_name][model_name]
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.2f}")
        
        return grid_search.best_estimator_
    
    def get_feature_importance(self, X_train, y_train, task_name, model_name):
        """Get feature importance from tree-based models"""
        model = self.models[task_name][model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[f"{task_name}_{model_name}"] = importance_df
            return importance_df
        else:
            print(f"Feature importance not available for {model_name}")
            return None
    
    def plot_model_comparison(self, task_name):
        """Plot comparison of different models"""
        if task_name not in self.model_scores:
            print(f"No scores available for {task_name}")
            return
        
        scores = self.model_scores[task_name]
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        
        if not valid_scores:
            print(f"No valid scores for {task_name}")
            return
        
        # Prepare data for plotting
        model_names = list(valid_scores.keys())
        mae_scores = [valid_scores[name]['mae'] for name in model_names]
        r2_scores = [valid_scores[name]['r2'] for name in model_names]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MAE comparison
        ax1.bar(model_names, mae_scores, color='skyblue', alpha=0.7)
        ax1.set_title(f'Model Comparison - MAE ({task_name})')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.tick_params(axis='x', rotation=45)
        
        # R² comparison
        ax2.bar(model_names, r2_scores, color='lightcoral', alpha=0.7)
        ax2.set_title(f'Model Comparison - R² Score ({task_name})')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'results/model_comparison_{task_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, task_name, model_name, top_n=20):
        """Plot feature importance"""
        key = f"{task_name}_{model_name}"
        if key not in self.feature_importance:
            print(f"Feature importance not available for {key}")
            return
        
        importance_df = self.feature_importance[key].head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name} ({task_name})')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'results/feature_importance_{key}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save trained models"""
        for task_name, best_model in self.best_models.items():
            model_path = f"{DATA_CONFIG['model_save_path']}best_{task_name}_model.pkl"
            joblib.dump(best_model['model'], model_path)
            print(f"Best {task_name} model saved to {model_path}")
    
    def load_models(self):
        """Load trained models"""
        for task_name in ['travel_time', 'stop_duration']:
            model_path = f"{DATA_CONFIG['model_save_path']}best_{task_name}_model.pkl"
            if os.path.exists(model_path):
                self.best_models[task_name] = {
                    'model': joblib.load(model_path),
                    'name': 'loaded_model'
                }
                print(f"Loaded {task_name} model from {model_path}")
    
    def predict(self, X, task_name):
        """Make predictions using the best model for a task"""
        if task_name not in self.best_models:
            print(f"No trained model available for {task_name}")
            return None
        
        model = self.best_models[task_name]['model']
        return model.predict(X)
    
    def evaluate_model(self, X_test, y_test, task_name):
        """Evaluate the best model on test data"""
        if task_name not in self.best_models:
            print(f"No trained model available for {task_name}")
            return None
        
        model = self.best_models[task_name]['model']
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nFinal Evaluation - {task_name}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.3f}")
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }

def main():
    """Main function to train and evaluate models"""
    print("Starting ML model training and evaluation...")
    
    # Load preprocessed data
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/train_data.csv')
    
    # Prepare features
    X, y = preprocessor.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Initialize predictor
    predictor = TrainArrivalPredictor()
    predictor.initialize_models()
    
    # Train models for travel time prediction
    y_travel_train = y_train['actual_travel_time']
    y_travel_test = y_test['actual_travel_time']
    
    predictor.train_models(X_train, y_travel_train, X_test, y_travel_test, 'travel_time')
    
    # Train models for stop duration prediction
    y_stop_train = y_train['actual_stop_time']
    y_stop_test = y_test['actual_stop_time']
    
    predictor.train_models(X_train, y_stop_train, X_test, y_stop_test, 'stop_duration')
    
    # Find best models
    predictor.find_best_models()
    
    # Get feature importance for best models
    for task_name in ['travel_time', 'stop_duration']:
        if task_name in predictor.best_models:
            best_model_name = predictor.best_models[task_name]['name']
            predictor.get_feature_importance(X_train, y_travel_train if task_name == 'travel_time' else y_stop_train, task_name, best_model_name)
    
    # Plot comparisons
    predictor.plot_model_comparison('travel_time')
    predictor.plot_model_comparison('stop_duration')
    
    # Plot feature importance
    for task_name in ['travel_time', 'stop_duration']:
        if task_name in predictor.best_models:
            best_model_name = predictor.best_models[task_name]['name']
            predictor.plot_feature_importance(task_name, best_model_name)
    
    # Save models
    predictor.save_models()
    
    # Final evaluation
    predictor.evaluate_model(X_test, y_travel_test, 'travel_time')
    predictor.evaluate_model(X_test, y_stop_test, 'stop_duration')
    
    print("\nModel training and evaluation completed!")

if __name__ == "__main__":
    main()
