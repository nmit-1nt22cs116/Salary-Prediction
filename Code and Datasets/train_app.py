"""
Flexible training script that works with:
1. Your 6k dataset with salary information (primary)
2. Optional: 54k structured resume dataset (for augmentation)
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import List, Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')


class FlexibleSalaryPredictor:
    """
    Flexible trainer that works with multiple dataset formats
    """

    def __init__(self):
        self.combined_df = None
        self.tfidf = None
        self.model = None

    def load_6k_dataset(self, csv_path: str, resume_col: str, salary_col: str) -> pd.DataFrame:
        """
        Load the 6k dataset with salary information
        """
        print("=" * 80)
        print("📂 LOADING 6K DATASET (WITH SALARY)")
        print("=" * 80)

        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None

            for enc in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    print(f"✅ Loaded successfully with encoding: {enc}")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                print("❌ Failed to load dataset with any encoding")
                return None

            print(f"\n📊 Dataset Info:")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {list(df.columns)}")

            # Check for required columns
            if resume_col not in df.columns:
                print(f"\n⚠️  Resume column '{resume_col}' not found!")
                print(f"   Available columns: {list(df.columns)}")

                # Try to find similar columns
                resume_like = [col for col in df.columns if any(
                    keyword in col.lower()
                    for keyword in ['resume', 'text', 'description', 'cv', 'content', 'summary']
                )]
                if resume_like:
                    print(f"   💡 Possible resume columns: {resume_like}")
                return None

            if salary_col not in df.columns:
                print(f"\n⚠️  Salary column '{salary_col}' not found!")
                print(f"   Available columns: {list(df.columns)}")

                # Try to find similar columns
                salary_like = [col for col in df.columns if any(
                    keyword in col.lower()
                    for keyword in ['salary', 'pay', 'wage', 'compensation', 'income']
                )]
                if salary_like:
                    print(f"   💡 Possible salary columns: {salary_like}")
                return None

            # Select required columns
            df_subset = df[[resume_col, salary_col]].copy()
            df_subset.columns = ['resume_text', 'salary']  # Standardize column names
            df_subset['data_source'] = '6k_dataset'

            print(f"\n✅ Successfully loaded 6k dataset")
            print(f"   Resume column: '{resume_col}'")
            print(f"   Salary column: '{salary_col}'")

            return df_subset

        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def load_54k_structured_dataset(self, data_folder: str) -> Optional[pd.DataFrame]:
        """
        Optionally load the 54k structured dataset (without salary)
        This can be used for transfer learning or to expand vocabulary
        """
        print("\n" + "=" * 80)
        print("📂 LOADING 54K STRUCTURED DATASET (OPTIONAL)")
        print("=" * 80)

        try:
            data_folder = Path(data_folder)

            csv_files = {
                'people': '01_people.csv',
                'education': '03_education.csv',
                'experience': '04_experience.csv',
                'person_skills': '05_person_skills.csv',
                'skills': '06_skills.csv'
            }

            dataframes = {}

            # Load CSV files
            for key, filename in csv_files.items():
                filepath = data_folder / filename
                if filepath.exists():
                    df = pd.read_csv(filepath, encoding='utf-8')
                    dataframes[key] = df
                    print(f"✅ Loaded {filename}: {len(df):,} rows")

            if not dataframes:
                print("⚠️  No files loaded from 54k dataset")
                return None

            # Merge datasets
            merged_df = dataframes.get('people', pd.DataFrame())

            if 'education' in dataframes:
                education_agg = dataframes['education'].groupby('person_id').agg({
                    'degree': lambda x: ' '.join(x.astype(str)),
                    'field': lambda x: ' '.join(x.astype(str)),
                }).reset_index()
                merged_df = merged_df.merge(education_agg, left_on='id', right_on='person_id', how='left')

            if 'experience' in dataframes:
                experience_agg = dataframes['experience'].groupby('person_id').agg({
                    'title': lambda x: ' '.join(x.astype(str)),
                    'description': lambda x: ' '.join(x.astype(str))
                }).reset_index()
                merged_df = merged_df.merge(experience_agg, left_on='id', right_on='person_id', how='left')

            if 'person_skills' in dataframes and 'skills' in dataframes:
                skills_merged = dataframes['person_skills'].merge(
                    dataframes['skills'],
                    left_on='skill_id',
                    right_on='id',
                    suffixes=('_person', '_skill')
                )
                skills_agg = skills_merged.groupby('person_id').agg({
                    'name': lambda x: ' '.join(x.astype(str))
                }).reset_index()
                merged_df = merged_df.merge(skills_agg, left_on='id', right_on='person_id', how='left')

            # Create resume text
            text_cols = [col for col in merged_df.columns if col in [
                'first_name', 'last_name', 'degree', 'field', 'title', 'description', 'name'
            ]]

            merged_df['resume_text'] = merged_df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
            merged_df['resume_text'] = merged_df['resume_text'].str.strip().str.replace(r'\s+', ' ', regex=True)

            # No salary for this dataset
            merged_df['salary'] = np.nan
            merged_df['data_source'] = '54k_structured'

            result_df = merged_df[['resume_text', 'salary', 'data_source']].copy()

            print(f"\n✅ Loaded 54k structured dataset: {len(result_df):,} rows")
            return result_df

        except Exception as e:
            print(f"⚠️  Could not load 54k dataset: {str(e)}")
            return None

    def combine_datasets(self, df_6k: pd.DataFrame, df_54k: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Combine the 6k dataset with optional 54k dataset
        """
        print("\n" + "=" * 80)
        print("🔗 COMBINING DATASETS")
        print("=" * 80)

        if df_54k is not None and len(df_54k) > 0:
            # Only use 54k data that might help (we'll use it for vocabulary expansion)
            # But we can't use it for training since it has no salary
            print("📊 Strategy: Using 6k for training, 54k for vocabulary enrichment")

            # We'll just use the 6k dataset for training
            # The 54k can help build a richer TF-IDF vocabulary
            all_text = pd.concat([df_6k['resume_text'], df_54k['resume_text'].dropna()])
            print(f"   Total text samples for TF-IDF: {len(all_text):,}")

            # But for training, only use 6k with salary
            self.combined_df = df_6k.copy()
            self.all_text_for_tfidf = all_text

            print(f"\n✅ Training dataset: {len(self.combined_df):,} rows (6k with salary)")

        else:
            print("📊 Using only 6k dataset for training")
            self.combined_df = df_6k.copy()
            self.all_text_for_tfidf = df_6k['resume_text']

            print(f"✅ Training dataset: {len(self.combined_df):,} rows")

        return self.combined_df

    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the dataset
        """
        print("\n" + "=" * 80)
        print("🧹 CLEANING DATA")
        print("=" * 80)

        df = self.combined_df.copy()
        initial_rows = len(df)

        # Remove null values
        print(f"\n1. Removing null values...")
        df = df.dropna(subset=['resume_text', 'salary'])
        print(f"   Removed {initial_rows - len(df):,} rows with null values")

        # Remove empty resume texts
        print(f"\n2. Removing empty/short resume texts...")
        before = len(df)
        df = df[df['resume_text'].str.strip().str.len() > 20]
        print(f"   Removed {before - len(df):,} rows with short resumes")

        # Clean salary values
        print(f"\n3. Cleaning salary values...")
        before = len(df)
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        df = df[df['salary'] > 0]
        print(f"   Removed {before - len(df):,} rows with invalid salaries")

        # Remove outliers
        print(f"\n4. Removing salary outliers (IQR method)...")
        Q1 = df['salary'].quantile(0.25)
        Q3 = df['salary'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        before = len(df)
        df = df[(df['salary'] >= lower_bound) & (df['salary'] <= upper_bound)]
        print(f"   Removed {before - len(df):,} outliers")
        print(f"   Salary range: ${df['salary'].min():,.2f} - ${df['salary'].max():,.2f}")
        print(f"   Mean salary: ${df['salary'].mean():,.2f}")
        print(f"   Median salary: ${df['salary'].median():,.2f}")

        print(f"\n✅ Final cleaned dataset: {len(df):,} rows ({(len(df) / initial_rows) * 100:.1f}% retained)")

        self.combined_df = df
        return df

    def create_tfidf_features(self, max_features: int = 300, use_enhanced_vocab: bool = True) -> tuple:
        """
        Create TF-IDF features
        """
        print("\n" + "=" * 80)
        print("🔤 CREATING TF-IDF FEATURES")
        print("=" * 80)

        # For vocabulary building, use all text if available
        if use_enhanced_vocab and hasattr(self, 'all_text_for_tfidf'):
            print("📚 Building vocabulary from both datasets...")
            vocab_text = self.all_text_for_tfidf.values
        else:
            vocab_text = self.combined_df['resume_text'].values

        # For actual training, use only data with salary
        X_text = self.combined_df['resume_text'].values
        y = self.combined_df['salary'].values

        # Create TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )

        # Fit on all text for vocabulary
        if use_enhanced_vocab and hasattr(self, 'all_text_for_tfidf'):
            print(f"   Fitting on {len(vocab_text):,} texts for vocabulary...")
            self.tfidf.fit(vocab_text)
            # Transform only the training data
            X_tfidf = self.tfidf.transform(X_text)
        else:
            print(f"   Fitting on {len(X_text):,} texts...")
            X_tfidf = self.tfidf.fit_transform(X_text)

        print(f"\n✅ TF-IDF features created:")
        print(f"   Shape: {X_tfidf.shape}")
        print(f"   Features: {X_tfidf.shape[1]}")
        print(f"   Sparsity: {(1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100:.2f}%")

        return X_tfidf, y

    def train_models(self, X, y) -> Dict[str, Any]:
        """
        Train multiple models and select the best one
        """
        print("\n" + "=" * 80)
        print("🤖 TRAINING MODELS")
        print("=" * 80)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\nTrain set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")

        # Define models
        models = {
            'LinearRegression': LinearRegression(),
            'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'RandomForest': RandomForestRegressor(
                random_state=42,
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                n_jobs=-1
            )
        }

        results = {}

        print("\nTraining models...\n")
        for name, model in models.items():
            print(f"📊 {name}:")
            try:
                # Train
                model.fit(X_train, y_train)

                # Predict
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Evaluate
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                mae = mean_absolute_error(y_test, y_test_pred)

                results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'rmse': rmse,
                    'mae': mae
                }

                print(f"   Train R²: {train_r2:.4f}")
                print(f"   Test R²:  {test_r2:.4f}")
                print(f"   RMSE:     ${rmse:,.2f}")
                print(f"   MAE:      ${mae:,.2f}")
                print(f"   ✅ Success\n")

            except Exception as e:
                print(f"   ❌ Failed: {str(e)}\n")
                results[name] = {'error': str(e)}

        # Select best model
        best_name = max(
            [name for name, res in results.items() if 'test_r2' in res],
            key=lambda name: results[name]['test_r2']
        )

        print("=" * 80)
        print(f"🏆 BEST MODEL: {best_name}")
        print(f"   Test R²: {results[best_name]['test_r2']:.4f}")
        print(f"   RMSE: ${results[best_name]['rmse']:,.2f}")
        print("=" * 80)

        self.model = results[best_name]['model']

        return {
            'best_model_name': best_name,
            'best_model': results[best_name]['model'],
            'all_results': results
        }

    def save_model(self, output_path: str, training_results: Dict) -> None:
        """
        Save complete artifact with TF-IDF pipeline
        """
        print("\n" + "=" * 80)
        print("💾 SAVING MODEL")
        print("=" * 80)

        artifact = {
            'model': self.model,
            'pipeline': {
                'tfidf': self.tfidf
            },
            'feature_names': self.tfidf.get_feature_names_out().tolist(),
            'metadata': {
                'model_type': training_results['best_model_name'],
                'n_features': self.tfidf.max_features,
                'n_training_samples': len(self.combined_df),
                'data_sources': self.combined_df['data_source'].unique().tolist(),
                'train_r2': training_results['all_results'][training_results['best_model_name']]['train_r2'],
                'test_r2': training_results['all_results'][training_results['best_model_name']]['test_r2'],
                'rmse': training_results['all_results'][training_results['best_model_name']]['rmse'],
                'mae': training_results['all_results'][training_results['best_model_name']]['mae'],
                'salary_range': {
                    'min': float(self.combined_df['salary'].min()),
                    'max': float(self.combined_df['salary'].max()),
                    'mean': float(self.combined_df['salary'].mean()),
                    'median': float(self.combined_df['salary'].median())
                },
                'training_date': pd.Timestamp.now().isoformat(),
                'tfidf_params': {
                    'max_features': self.tfidf.max_features,
                    'ngram_range': self.tfidf.ngram_range,
                    'min_df': self.tfidf.min_df,
                    'max_df': self.tfidf.max_df
                }
            }
        }

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save
        joblib.dump(artifact, output_path)

        print(f"\n✅ Model saved to: {output_path}")
        print(f"\nArtifact contents:")
        print(f"  ✓ model: {type(artifact['model']).__name__}")
        print(f"  ✓ pipeline: {list(artifact['pipeline'].keys())}")
        print(f"  ✓ feature_names: {len(artifact['feature_names'])} features")
        print(f"  ✓ metadata: Complete")
        print(f"\n🎯 TF-IDF pipeline included: YES")
        print(f"📊 Model ready for deployment!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🚀 " * 30)
    print("FLEXIBLE SALARY PREDICTION MODEL TRAINER")
    print("🚀 " * 30)

    # ============================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ============================================================

    # Your 6k dataset with salary information (PRIMARY)
    DATASET_6K_PATH = 'data/salary_data_large.csv'  # ⚠️ UPDATE THIS PATH
    RESUME_COLUMN_6K = 'Job Title'  # ⚠️ UPDATE: Column with resume text
    SALARY_COLUMN_6K = 'Salary'  # ⚠️ UPDATE: Column with salary

    # Optional: 54k structured dataset (for vocabulary enrichment)
    USE_54K_DATASET = False  # Set to True if you want to use it
    DATASET_54K_FOLDER = '../data/54k Resume dataset (structured)'

    # Output path
    OUTPUT_PATH = 'artifacts/salary_model.pkl'

    # TF-IDF features
    MAX_FEATURES = 300

    # ============================================================

    try:
        predictor = FlexibleSalaryPredictor()

        # Load 6k dataset (PRIMARY - has salary)
        print("\n" + "=" * 80)
        print("STEP 1: Loading 6k dataset with salary")
        print("=" * 80)
        df_6k = predictor.load_6k_dataset(
            csv_path=DATASET_6K_PATH,
            resume_col=RESUME_COLUMN_6K,
            salary_col=SALARY_COLUMN_6K
        )

        if df_6k is None or len(df_6k) == 0:
            print("\n❌ Failed to load 6k dataset. Please check the path and column names.")
            print("\nTo find the correct column names:")
            print("  1. Open your CSV file")
            print("  2. Look at the first row (headers)")
            print("  3. Update RESUME_COLUMN_6K and SALARY_COLUMN_6K variables")
            exit(1)

        # Optionally load 54k dataset (for vocabulary enrichment)
        df_54k = None
        if USE_54K_DATASET:
            print("\n" + "=" * 80)
            print("STEP 2: Loading 54k dataset (optional - for vocabulary)")
            print("=" * 80)
            df_54k = predictor.load_54k_structured_dataset(DATASET_54K_FOLDER)

        # Combine datasets
        predictor.combine_datasets(df_6k, df_54k)

        # Clean data
        predictor.clean_data()

        # Create TF-IDF features
        X, y = predictor.create_tfidf_features(
            max_features=MAX_FEATURES,
            use_enhanced_vocab=(df_54k is not None)
        )

        # Train models
        results = predictor.train_models(X, y)

        # Save model
        predictor.save_model(OUTPUT_PATH, results)

        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n✅ Your model is ready at: {OUTPUT_PATH}")
        print(f"✅ You can now use it in your Streamlit app!")
        print(f"\n📊 Training Summary:")
        print(f"   • Trained on: {len(predictor.combined_df):,} resumes with salary")
        print(f"   • Model: {results['best_model_name']}")
        print(f"   • Test R²: {results['all_results'][results['best_model_name']]['test_r2']:.4f}")
        print(f"   • RMSE: ${results['all_results'][results['best_model_name']]['rmse']:,.2f}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TRAINING FAILED")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Check that DATASET_6K_PATH points to your 6k CSV file")
        print("  2. Verify RESUME_COLUMN_6K and SALARY_COLUMN_6K match your CSV headers")
        print("  3. Ensure the CSV file has both resume text and salary columns")
        print("  4. Check file encoding (try opening in Excel/text editor)")
        import traceback

        traceback.print_exc()