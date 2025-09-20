"""
Simplified ML Model Training for Train Arrival Time Prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from data_preprocessing import DataPreprocessor

class SimpleTrainPredictor:
    def __init__(self):
        self.models = {}
        self.scores = {}
        self.best_models = {}
        
    def train_models(self, X_train, y_train, X_test, y_test, task_name):
        """Train simplified models for a specific task"""
        print(f"\nTraining models for {task_name} prediction...")
        
        # Define models
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=-1
            )
        }
        
        scores = {}
        
        for model_name, model in models.items():
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
                
                scores[model_name] = {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'model': model
                }
                
                print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
                scores[model_name] = None
        
        self.scores[task_name] = scores
        return scores
    
    def find_best_models(self):
        """Find the best performing models for each task"""
        for task_name in ['travel_time', 'stop_duration']:
            if task_name in self.scores:
                scores = self.scores[task_name]
                valid_scores = {k: v for k, v in scores.items() if v is not None}
                
                if valid_scores:
                    # Find best model based on MAE (lower is better)
                    best_model = min(valid_scores.items(), key=lambda x: x[1]['mae'])
                    self.best_models[task_name] = {
                        'name': best_model[0],
                        'model': best_model[1]['model'],
                        'scores': best_model[1]
                    }
                    print(f"Best model for {task_name}: {best_model[0]} (MAE: {best_model[1]['mae']:.2f})")
    
    def plot_model_comparison(self, task_name):
        """Plot comparison of different models"""
        if task_name not in self.scores:
            print(f"No scores available for {task_name}")
            return
        
        scores = self.scores[task_name]
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
    
    def get_feature_importance(self, X_train, task_name, model_name):
        """Get feature importance from tree-based models"""
        if task_name not in self.scores or model_name not in self.scores[task_name]:
            return None
            
        model = self.scores[task_name][model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print(f"Feature importance not available for {model_name}")
            return None
    
    def plot_feature_importance(self, X_train, task_name, model_name, top_n=20):
        """Plot feature importance"""
        importance_df = self.get_feature_importance(X_train, task_name, model_name)
        
        if importance_df is None:
            return
        
        importance_df = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name} ({task_name})')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'results/feature_importance_{task_name}_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """Save trained models"""
        for task_name, best_model in self.best_models.items():
            model_path = f"models/best_{task_name}_model.pkl"
            joblib.dump(best_model['model'], model_path)
            print(f"Best {task_name} model saved to {model_path}")
    
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
    print("Starting simplified ML model training and evaluation...")
    
    # Load preprocessed data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/train_data.csv')
    
    # Prepare features
    X, y = preprocessor.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Initialize predictor
    predictor = SimpleTrainPredictor()
    
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
    
    # Plot comparisons
    predictor.plot_model_comparison('travel_time')
    predictor.plot_model_comparison('stop_duration')
    
    # Plot feature importance for best models
    for task_name in ['travel_time', 'stop_duration']:
        if task_name in predictor.best_models:
            best_model_name = predictor.best_models[task_name]['name']
            predictor.plot_feature_importance(X_train, task_name, best_model_name)
    
    # Save models
    predictor.save_models()
    
    # Final evaluation
    predictor.evaluate_model(X_test, y_travel_test, 'travel_time')
    predictor.evaluate_model(X_test, y_stop_test, 'stop_duration')
    
    print("\nModel training and evaluation completed!")

if __name__ == "__main__":
    main()
