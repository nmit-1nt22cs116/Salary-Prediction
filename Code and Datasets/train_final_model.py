"""
Final Production Model Training with Enhanced 54K Dataset
Using ALL available features for maximum accuracy
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class FinalModelTrainer:
    """Final production model with all features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_names = None
        
    def load_and_engineer_features(self, csv_path='data/enhanced_54k_dataset.csv'):
        """Load data and engineer comprehensive features"""
        print("=" * 80)
        print("📂 LOADING ENHANCED DATASET")
        print("=" * 80)
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df):,} samples")
        
        # Clean data
        df = df.dropna(subset=['Salary'])
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
        df = df[df['Salary'] > 0]
        
        print(f"After cleaning: {len(df):,} samples")
        
        # Feature Engineering
        print("\n🔧 Engineering Features...")
        
        # 1. Encode categorical variables
        df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'].fillna('Unknown'))
        
        # 2. Education level mapping
        education_map = {
            "High School": 1,
            "Bachelor's": 3,
            "Master's": 4,
            "PhD": 5
        }
        df['Education_level'] = df['Education Level'].map(education_map).fillna(3)
        
        # 3. Job level from title
        df['is_senior'] = df['Job Title'].str.lower().str.contains('senior|sr|lead|principal', na=False).astype(int)
        df['is_manager'] = df['Job Title'].str.lower().str.contains('manager|director|head|vp', na=False).astype(int)
        df['is_executive'] = df['Job Title'].str.lower().str.contains('ceo|cto|cfo|president|executive', na=False).astype(int)
        df['is_junior'] = df['Job Title'].str.lower().str.contains('junior|jr|entry|intern', na=False).astype(int)
        
        # 4. Job category
        df['is_tech'] = df['Job Title'].str.lower().str.contains('software|developer|engineer|data|analyst|scientist', na=False).astype(int)
        df['is_sales'] = df['Job Title'].str.lower().str.contains('sales|account', na=False).astype(int)
        df['is_marketing'] = df['Job Title'].str.lower().str.contains('marketing', na=False).astype(int)
        df['is_hr'] = df['Job Title'].str.lower().str.contains('hr|human|recruiter', na=False).astype(int)
        
        # 5. Experience features
        df['Years of Experience'] = pd.to_numeric(df['Years of Experience'], errors='coerce').fillna(0)
        df['experience_squared'] = df['Years of Experience'] ** 2
        df['experience_education'] = df['Years of Experience'] * df['Education_level']
        
        # 6. Age features
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(df['Age'].median())
        df['age_experience_ratio'] = df['Age'] / (df['Years of Experience'] + 1)
        
        # 7. Interaction features
        df['senior_tech'] = df['is_senior'] * df['is_tech']
        df['manager_experience'] = df['is_manager'] * df['Years of Experience']
        df['education_tech'] = df['Education_level'] * df['is_tech']
        
        # 8. Binned features
        df['experience_bin'] = pd.cut(df['Years of Experience'], 
                                       bins=[-1, 2, 5, 10, 20, 50],
                                       labels=[1, 2, 3, 4, 5]).astype(int)
        
        df['age_bin'] = pd.cut(df['Age'], 
                               bins=[0, 25, 35, 45, 55, 100],
                               labels=[1, 2, 3, 4, 5]).astype(int)
        
        # Select features for training
        feature_cols = [
            'Years of Experience', 'Education_level', 'Age', 'Gender_encoded',
            'is_senior', 'is_manager', 'is_executive', 'is_junior',
            'is_tech', 'is_sales', 'is_marketing', 'is_hr',
            'experience_squared', 'experience_education', 'age_experience_ratio',
            'senior_tech', 'manager_experience', 'education_tech',
            'experience_bin', 'age_bin'
        ]
        
        X = df[feature_cols].fillna(0)
        y = df['Salary'].values
        
        self.feature_names = feature_cols
        
        print(f"\n✅ Features engineered: {len(feature_cols)} features")
        print(f"   Salary range: ${y.min():,.0f} - ${y.max():,.0f}")
        print(f"   Mean salary: ${y.mean():,.2f}")
        
        return X, y, df
    
    def train_optimized_models(self, X_train, X_test, y_train, y_test):
        """Train multiple optimized models with comprehensive metrics"""
        print("\n" + "=" * 80)
        print("🚀 TRAINING OPTIMIZED MODELS")
        print("=" * 80)
        
        models = {}
        
        # 1. XGBoost - Highly tuned
        print("\n1. Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train, 
                      eval_set=[(X_test, y_test)],
                      verbose=False)
        
        # Calculate train metrics
        xgb_train_pred = xgb_model.predict(X_train)
        xgb_train_r2 = r2_score(y_train, xgb_train_pred)
        
        # Calculate test metrics
        xgb_test_pred = xgb_model.predict(X_test)
        xgb_test_r2 = r2_score(y_test, xgb_test_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
        xgb_mae = mean_absolute_error(y_test, xgb_test_pred)
        xgb_mape = np.mean(np.abs((y_test - xgb_test_pred) / y_test)) * 100
        
        models['XGBoost'] = {
            'model': xgb_model,
            'train_r2': xgb_train_r2,
            'test_r2': xgb_test_r2,
            'rmse': xgb_rmse,
            'mae': xgb_mae,
            'mape': xgb_mape,
            'train_pred': xgb_train_pred,
            'test_pred': xgb_test_pred
        }
        print(f"   XGBoost Train R²: {xgb_train_r2:.4f} ({xgb_train_r2*100:.2f}%) | Test R²: {xgb_test_r2:.4f} ({xgb_test_r2*100:.2f}%)")
        print(f"   RMSE: ${xgb_rmse:,.2f} | MAE: ${xgb_mae:,.2f} | MAPE: {xgb_mape:.2f}%")
        
        # 2. Gradient Boosting
        print("\n2. Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        # Calculate train metrics
        gb_train_pred = gb_model.predict(X_train)
        gb_train_r2 = r2_score(y_train, gb_train_pred)
        
        # Calculate test metrics
        gb_test_pred = gb_model.predict(X_test)
        gb_test_r2 = r2_score(y_test, gb_test_pred)
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
        gb_mae = mean_absolute_error(y_test, gb_test_pred)
        gb_mape = np.mean(np.abs((y_test - gb_test_pred) / y_test)) * 100
        
        models['GradientBoosting'] = {
            'model': gb_model,
            'train_r2': gb_train_r2,
            'test_r2': gb_test_r2,
            'rmse': gb_rmse,
            'mae': gb_mae,
            'mape': gb_mape,
            'train_pred': gb_train_pred,
            'test_pred': gb_test_pred
        }
        print(f"   Gradient Boosting Train R²: {gb_train_r2:.4f} ({gb_train_r2*100:.2f}%) | Test R²: {gb_test_r2:.4f} ({gb_test_r2*100:.2f}%)")
        print(f"   RMSE: ${gb_rmse:,.2f} | MAE: ${gb_mae:,.2f} | MAPE: {gb_mape:.2f}%")
        
        # 3. Random Forest
        print("\n3. Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Calculate train metrics
        rf_train_pred = rf_model.predict(X_train)
        rf_train_r2 = r2_score(y_train, rf_train_pred)
        
        # Calculate test metrics
        rf_test_pred = rf_model.predict(X_test)
        rf_test_r2 = r2_score(y_test, rf_test_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
        rf_mae = mean_absolute_error(y_test, rf_test_pred)
        rf_mape = np.mean(np.abs((y_test - rf_test_pred) / y_test)) * 100
        
        models['RandomForest'] = {
            'model': rf_model,
            'train_r2': rf_train_r2,
            'test_r2': rf_test_r2,
            'rmse': rf_rmse,
            'mae': rf_mae,
            'mape': rf_mape,
            'train_pred': rf_train_pred,
            'test_pred': rf_test_pred
        }
        print(f"   Random Forest Train R²: {rf_train_r2:.4f} ({rf_train_r2*100:.2f}%) | Test R²: {rf_test_r2:.4f} ({rf_test_r2*100:.2f}%)")
        print(f"   RMSE: ${rf_rmse:,.2f} | MAE: ${rf_mae:,.2f} | MAPE: {rf_mape:.2f}%")
        
        # 4. Ensemble (Weighted Average)
        print("\n4. Creating Weighted Ensemble...")
        # Weight by test R² score
        total_r2 = xgb_test_r2 + gb_test_r2 + rf_test_r2
        w_xgb = xgb_test_r2 / total_r2
        w_gb = gb_test_r2 / total_r2
        w_rf = rf_test_r2 / total_r2
        
        # Train predictions
        ensemble_train_pred = w_xgb * xgb_train_pred + w_gb * gb_train_pred + w_rf * rf_train_pred
        ensemble_train_r2 = r2_score(y_train, ensemble_train_pred)
        
        # Test predictions
        ensemble_test_pred = w_xgb * xgb_test_pred + w_gb * gb_test_pred + w_rf * rf_test_pred
        ensemble_test_r2 = r2_score(y_test, ensemble_test_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_pred))
        ensemble_mae = mean_absolute_error(y_test, ensemble_test_pred)
        ensemble_mape = np.mean(np.abs((y_test - ensemble_test_pred) / y_test)) * 100
        
        print(f"   Weights: XGB={w_xgb:.3f}, GB={w_gb:.3f}, RF={w_rf:.3f}")
        
        # Store ensemble
        models['Ensemble'] = {
            'model': {
                'xgb': xgb_model,
                'gb': gb_model,
                'rf': rf_model,
                'weights': (w_xgb, w_gb, w_rf)
            },
            'train_r2': ensemble_train_r2,
            'test_r2': ensemble_test_r2,
            'rmse': ensemble_rmse,
            'mae': ensemble_mae,
            'mape': ensemble_mape,
            'train_pred': ensemble_train_pred,
            'test_pred': ensemble_test_pred
        }
        print(f"   Ensemble Train R²: {ensemble_train_r2:.4f} ({ensemble_train_r2*100:.2f}%) | Test R²: {ensemble_test_r2:.4f} ({ensemble_test_r2*100:.2f}%)")
        print(f"   RMSE: ${ensemble_rmse:,.2f} | MAE: ${ensemble_mae:,.2f} | MAPE: {ensemble_mape:.2f}%")
        
        return models
    
    def train_final_model(self):
        """Train the final production model"""
        # Load data
        X, y, df = self.load_and_engineer_features()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Train models
        models = self.train_optimized_models(X_train, X_test, y_train, y_test)
        
        # Select best model based on test R²
        best_name = max(models.items(), key=lambda x: x[1]['test_r2'])[0]
        best_model_data = models[best_name]
        self.best_model = best_model_data['model']
        
        print("\n" + "=" * 80)
        print(f"🏆 BEST MODEL: {best_name}")
        print(f"   Train R²: {best_model_data['train_r2']:.4f} ({best_model_data['train_r2']*100:.2f}%)")
        print(f"   Test R²: {best_model_data['test_r2']:.4f} ({best_model_data['test_r2']*100:.2f}%)")
        print(f"   RMSE: ${best_model_data['rmse']:,.2f}")
        print(f"   MAE: ${best_model_data['mae']:,.2f}")
        print(f"   MAPE: {best_model_data['mape']:.2f}%")
        print("=" * 80)
        
        # Print comparison table
        print("\n" + "=" * 80)
        print("📊 MODEL COMPARISON")
        print("=" * 80)
        print(f"{'Model':<20} {'Train R²':<12} {'Test R²':<12} {'RMSE':<15} {'MAE':<15} {'MAPE':<10}")
        print("-" * 80)
        for name, data in models.items():
            print(f"{name:<20} {data['train_r2']:.4f} ({data['train_r2']*100:5.2f}%) {data['test_r2']:.4f} ({data['test_r2']*100:5.2f}%) ${data['rmse']:>12,.2f} ${data['mae']:>12,.2f} {data['mape']:>6.2f}%")
        print("=" * 80)
        
        results = {
            'model': self.best_model,
            'model_name': best_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'train_r2': best_model_data['train_r2'],
            'test_r2': best_model_data['test_r2'],
            'rmse': best_model_data['rmse'],
            'mae': best_model_data['mae'],
            'mape': best_model_data['mape'],
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': best_model_data['test_pred'],
            'all_models': models
        }
        
        return results
    
    def _predict(self, model, X, model_name):
        """Helper to predict with different model types"""
        if model_name == 'Ensemble':
            xgb_pred = model['xgb'].predict(X)
            gb_pred = model['gb'].predict(X)
            rf_pred = model['rf'].predict(X)
            w_xgb, w_gb, w_rf = model['weights']
            return w_xgb * xgb_pred + w_gb * gb_pred + w_rf * rf_pred
        else:
            return model.predict(X)
    
    def save_model(self, results, output_path='models/production_model.pkl'):
        """Save production model"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        artifact = {
            'model': results['model'],
            'scaler': results['scaler'],
            'feature_names': results['feature_names'],
            'metadata': {
                'model_name': results['model_name'],
                'train_r2': results['train_r2'],
                'test_r2': results['test_r2'],
                'rmse': results['rmse'],
                'mae': results['mae'],
                'mape': results['mape'],
                'accuracy_percentage': results['test_r2'] * 100,
                'features_count': len(results['feature_names'])
            }
        }
        
        joblib.dump(artifact, output_path)
        print(f"\n✅ Model saved to: {output_path}")
        return artifact


if __name__ == "__main__":
    print("\n" + "🎯 " * 30)
    print("FINAL PRODUCTION MODEL TRAINING")
    print("Using Enhanced 54K Dataset with ALL Features")
    print("🎯 " * 30)
    
    try:
        trainer = FinalModelTrainer()
        results = trainer.train_final_model()
        trainer.save_model(results)
        
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Final Results:")
        print(f"   Model: {results['model_name']}")
        print(f"   Train R²: {results['train_r2']:.4f} ({results['train_r2']*100:.2f}%)")
        print(f"   Test R²: {results['test_r2']:.4f} ({results['test_r2']*100:.2f}%)")
        print(f"   RMSE: ${results['rmse']:,.2f}")
        print(f"   MAE: ${results['mae']:,.2f}")
        print(f"   MAPE: {results['mape']:.2f}%")
        print(f"   Features: {len(results['feature_names'])}")
        
        # Accuracy interpretation
        accuracy_pct = results['test_r2'] * 100
        if accuracy_pct >= 90:
            print(f"\n✅ EXCELLENT: {accuracy_pct:.2f}% accuracy achieved!")
        elif accuracy_pct >= 80:
            print(f"\n✅ VERY GOOD: {accuracy_pct:.2f}% accuracy achieved!")
        elif accuracy_pct >= 70:
            print(f"\n✅ GOOD: {accuracy_pct:.2f}% accuracy achieved!")
        else:
            print(f"\n⚠️  Current accuracy: {accuracy_pct:.2f}%")
        
        print(f"\n💡 Model Performance:")
        print(f"   - Explains {accuracy_pct:.1f}% of salary variance")
        print(f"   - Average error: ${results['mae']:,.0f}")
        print(f"   - Percentage error: {results['mape']:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
