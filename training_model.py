#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np

import joblib

from datetime import datetime
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)



class MedicalDiagnosisCleaner:
    """
    Handles all data cleaning and preparation steps for medical diagnosis prediction.
    Transforms normalized database tables into ML-ready feature matrices.
    """

    def __init__(self):
        """Initialize the data cleaner."""
        self.feature_columns = []
        self.symptom_columns = []
        self.metadata = {}
        self.df_prepared = None

    def clean_and_prepare(self, df_cases: pd.DataFrame, df_symptoms: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline: clean and prepare data from normalized tables.

        Args:
            df_cases: DataFrame with patient cases
            df_symptoms: DataFrame with symptoms per case

        Returns:
            df_prepared: Cleaned DataFrame ready for ML
        """
        print("=" * 100)
        print("MEDICAL DIAGNOSIS DATA CLEANING PIPELINE")
        print("=" * 100)

        # Step 1: Validate input data
        print("\n[1/9] Validating input data...")
        self._validate_data(df_cases, df_symptoms)
        print("   ✓ Data validation passed")

        # Step 2: Clean cases table
        print("\n[2/9] Cleaning cases table...")
        df_cases_clean = self._clean_cases(df_cases)
        print(f"   ✓ Cases cleaned: {df_cases_clean.shape}")

        # Step 3: Clean symptoms table
        print("\n[3/9] Cleaning symptoms table...")
        df_symptoms_clean = self._clean_symptoms(df_symptoms)
        print(f"   ✓ Symptoms cleaned: {df_symptoms_clean.shape}")

        # Step 4: Pivot symptoms to wide format
        print("\n[4/9] Pivoting symptoms to wide format...")
        df_severity = self._pivot_symptoms(df_symptoms_clean)
        print(f"   ✓ Symptom matrix created: {df_severity.shape}")
        print(f"   ✓ Unique symptoms: {len(df_severity.columns)}")

        # Step 5: Engineer aggregate features
        print("\n[5/9] Engineering aggregate features...")
        df_features = self._engineer_features(df_symptoms_clean)
        print(f"   ✓ Aggregate features created: {len(df_features.columns) - 1}")

        # Step 6: Merge all data
        print("\n[6/9] Merging all features...")
        df_merged = self._merge_all(df_cases_clean, df_features, df_severity)
        print(f"   ✓ Merged dataset shape: {df_merged.shape}")

        # Step 7: Handle missing values
        print("\n[7/9] Handling missing values...")
        df_complete = self._handle_missing_values(df_merged)
        print(f"   ✓ Missing values handled: {df_complete.shape}")

        # Step 8: Encode categorical variables
        print("\n[8/9] Encoding categorical variables...")
        df_encoded = self._encode_categoricals(df_complete)
        print(f"   ✓ Encoding complete: {df_encoded.shape}")

        # Step 9: Remove useless fields
        print("\n[9/9] Removing useless fields...")
        df_prepared = self._remove_useless_fields(df_encoded)
        print(f"   ✓ Useless fields removed: {df_prepared.shape}")

        # Store the prepared dataframe
        self.df_prepared = df_prepared

        # Store metadata
        self._store_metadata(df_prepared, df_cases, df_symptoms)

        print("\n" + "=" * 100)
        print("DATA CLEANING COMPLETE ✅")
        print("=" * 100)
        self._print_summary()

        return df_prepared

    def _validate_data(self, df_cases: pd.DataFrame, df_symptoms: pd.DataFrame) -> None:
        """Validate that input data has required columns and structure."""
        # Check cases table
        required_case_cols = ['case_id', 'user_id', 'age', 'gender', 'timestamp', 'diagnosis', 'confidence']
        missing_case_cols = [col for col in required_case_cols if col not in df_cases.columns]
        if missing_case_cols:
            raise ValueError(f"Cases table missing columns: {missing_case_cols}")

        # Check symptoms table
        required_symptom_cols = ['case_id', 'symptom', 'severity', 'duration_hours']
        missing_symptom_cols = [col for col in required_symptom_cols if col not in df_symptoms.columns]
        if missing_symptom_cols:
            raise ValueError(f"Symptoms table missing columns: {missing_symptom_cols}")

        # Check for orphaned symptoms (symptoms without cases)
        orphaned = set(df_symptoms['case_id'].unique()) - set(df_cases['case_id'].unique())
        if orphaned:
            print(f"   ⚠ Warning: {len(orphaned)} orphaned symptom records found")

    def _clean_cases(self, df_cases: pd.DataFrame) -> pd.DataFrame:
        """Clean the cases table."""
        df = df_cases.copy()

        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['case_id'])
        if len(df) < initial_count:
            print(f"   ⚠ Removed {initial_count - len(df)} duplicate cases")

        # Clean age (remove invalid values)
        df = df[(df['age'] > 0) & (df['age'] < 120)]

        # Clean gender
        df['gender'] = df['gender'].str.upper()
        df = df[df['gender'].isin(['M', 'F'])]

        # Clean confidence
        df['confidence'] = df['confidence'].clip(0, 1)

        # Convert timestamp to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def _clean_symptoms(self, df_symptoms: pd.DataFrame) -> pd.DataFrame:
        """Clean the symptoms table."""
        df = df_symptoms.copy()

        # Remove duplicates (same case + same symptom)
        initial_count = len(df)
        df = df.drop_duplicates(subset=['case_id', 'symptom'])
        if len(df) < initial_count:
            print(f"   ⚠ Removed {initial_count - len(df)} duplicate symptoms")

        # Clean symptom names (lowercase, remove extra spaces)
        df['symptom'] = df['symptom'].str.lower().str.strip().str.replace(r'\s+', '_', regex=True)

        # Clean severity (1-10 range)
        df['severity'] = df['severity'].clip(1, 10)

        # Clean duration (remove negative values)
        df['duration_hours'] = df['duration_hours'].clip(0, None)

        # Clean body_location (lowercase, handle NaN)
        if 'body_location' in df.columns:
            df['body_location'] = df['body_location'].fillna('unknown')
            df['body_location'] = df['body_location'].str.lower().str.strip()

        return df

    def _pivot_symptoms(self, df_symptoms: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot symptoms to wide format: one column per symptom with severity values.
        """
        df_severity = df_symptoms.pivot_table(
            index='case_id',
            columns='symptom',
            values='severity',
            fill_value=0,
            aggfunc='max'  # If duplicate, take max severity
        )

        # Store symptom column names
        self.symptom_columns = list(df_severity.columns)

        return df_severity

    def _engineer_features(self, df_symptoms: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer aggregate features from symptoms.

        Returns:
            DataFrame with case_id and engineered features
        """
        features = {}

        # Group by case_id
        grouped = df_symptoms.groupby('case_id')

        # Feature 1: Total symptom count
        features['symptom_count'] = grouped.size()

        # Feature 2: Average severity
        features['avg_severity'] = grouped['severity'].mean()

        # Feature 3: Max severity
        features['max_severity'] = grouped['severity'].max()

        # Feature 4: Min severity
        features['min_severity'] = grouped['severity'].min()

        # Feature 5: Severity standard deviation
        features['std_severity'] = grouped['severity'].std().fillna(0)

        # Feature 6: Average duration
        features['avg_duration'] = grouped['duration_hours'].mean()

        # Feature 7: Max duration
        features['max_duration'] = grouped['duration_hours'].max()

        # Feature 8: Min duration
        features['min_duration'] = grouped['duration_hours'].min()

        # Feature 9: Total duration (sum of all symptoms)
        features['total_duration'] = grouped['duration_hours'].sum()

        # Feature 10: Has fever (binary)
        fever_symptoms = df_symptoms[df_symptoms['symptom'].str.contains('fever', case=False, na=False)]
        has_fever = fever_symptoms.groupby('case_id').size()
        features['has_fever'] = (has_fever > 0).astype(int)

        # Feature 11: Has pain (binary)
        pain_symptoms = df_symptoms[df_symptoms['symptom'].str.contains('pain|ache', case=False, na=False)]
        has_pain = pain_symptoms.groupby('case_id').size()
        features['has_pain'] = (has_pain > 0).astype(int)

        # Feature 12: Has nausea/vomiting (binary)
        gi_symptoms = df_symptoms[df_symptoms['symptom'].str.contains('nausea|vomit', case=False, na=False)]
        has_gi = gi_symptoms.groupby('case_id').size()
        features['has_gi_symptoms'] = (has_gi > 0).astype(int)

        # Feature 13: Has respiratory symptoms (binary)
        resp_symptoms = df_symptoms[df_symptoms['symptom'].str.contains('cough|breath|wheez', case=False, na=False)]
        has_resp = resp_symptoms.groupby('case_id').size()
        features['has_respiratory'] = (has_resp > 0).astype(int)

        # Combine all features into a DataFrame
        df_features = pd.DataFrame(features).reset_index()

        return df_features

    def _merge_all(self, df_cases: pd.DataFrame, df_features: pd.DataFrame, 
                   df_severity: pd.DataFrame) -> pd.DataFrame:
        """Merge cases, engineered features, and symptom matrix."""
        # Start with cases
        df = df_cases.copy()

        # Merge engineered features
        df = df.merge(df_features, on='case_id', how='left')

        # Merge symptom severity matrix
        df = df.merge(df_severity, left_on='case_id', right_index=True, how='left')

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()

        # Fill numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Fill categorical/string columns with 'unknown'
        # Use exclude to avoid the warning
        categorical_cols = df.select_dtypes(exclude=[np.number, 'datetime64']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')

        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()

        # Encode gender (M=0, F=1) - REPLACE the original gender column
        if 'gender' in df.columns:
            df['gender'] = (df['gender'] == 'F').astype(int)
            print("   ✓ Gender encoded: M=0, F=1")

        # Extract time features from timestamp
        if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            print("   ✓ Time features extracted: hour_of_day, day_of_week, month")

        return df

    def _remove_useless_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove fields not needed for ML training (case_id, user_id, timestamp, confidence).
        This is now part of the cleaning pipeline.
        """
        df = df.copy()

        # Fields to remove
        useless_fields = ['case_id', 'user_id', 'timestamp', 'confidence']
        fields_to_drop = [col for col in useless_fields if col in df.columns]

        if fields_to_drop:
            df = df.drop(columns=fields_to_drop)
            print(f"   ✓ Removed: {', '.join(fields_to_drop)}")

        return df

    def _store_metadata(self, df_prepared: pd.DataFrame, df_cases: pd.DataFrame, 
                       df_symptoms: pd.DataFrame) -> None:
        """Store metadata about the cleaning process."""
        self.metadata = {
            'original_cases': len(df_cases),
            'original_symptoms': len(df_symptoms),
            'final_cases': len(df_prepared),
            'total_features': df_prepared.shape[1],
            'symptom_features': len(self.symptom_columns),
            'unique_symptoms': self.symptom_columns,
            'diagnoses': list(df_prepared['diagnosis'].unique()),
            'n_diagnoses': df_prepared['diagnosis'].nunique(),
            'age_range': (df_prepared['age'].min(), df_prepared['age'].max()),
            'timestamp': datetime.now().isoformat()
        }

    def _print_summary(self) -> None:
        """Print summary of cleaned data."""
        print(f"\n📊 CLEANING SUMMARY:")
        print(f"   • Original cases:     {self.metadata['original_cases']}")
        print(f"   • Final cases:        {self.metadata['final_cases']}")
        print(f"   • Total features:     {self.metadata['total_features']}")
        print(f"   • Symptom features:   {self.metadata['symptom_features']}")
        print(f"   • Unique diagnoses:   {self.metadata['n_diagnoses']}")
        print(f"   • Age range:          {self.metadata['age_range'][0]}-{self.metadata['age_range'][1]} years")

    def get_feature_columns(self) -> List[str]:
        """
        Get list of feature columns (excludes diagnosis which is the target).

        Returns:
            List of feature column names
        """
        if self.df_prepared is None:
            raise ValueError("No data prepared yet. Call clean_and_prepare() first.")

        # Exclude only diagnosis (target variable)
        feature_cols = [col for col in self.df_prepared.columns if col != 'diagnosis']

        return feature_cols

    def get_X_y(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get feature matrix (X) and target vector (y).

        Returns:
            X: Features DataFrame
            y: Target Series (diagnosis - NOT encoded yet)
        """
        if self.df_prepared is None:
            raise ValueError("No data prepared yet. Call clean_and_prepare() first.")

        feature_cols = self.get_feature_columns()
        X = self.df_prepared[feature_cols]
        y = self.df_prepared['diagnosis']

        return X, y

    def save_prepared_data(self, filepath: str) -> None:
        """Save prepared data to CSV."""
        if self.df_prepared is None:
            raise ValueError("No data prepared yet. Call clean_and_prepare() first.")

        self.df_prepared.to_csv(filepath, index=False)
        print(f"\n💾 Prepared data saved to: {filepath}")

    def get_metadata(self) -> Dict:
        """Get metadata about the cleaning process."""
        return self.metadata


class MedicalDiagnosisPredictor:
    """
    Handles ML model training, evaluation, comparison, and prediction for medical diagnosis.
    Works with data prepared by MedicalDiagnosisCleaner.
    """

    def __init__(self):
        """Initialize the predictor."""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_history = []
        self.comparison_results = pd.DataFrame()

    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, 
                    random_state: int = 42) -> None:
        """
        Prepare data for training: encode labels, split, and scale.

        Args:
            X: Feature matrix (from cleaner.get_X_y())
            y: Target variable (diagnosis names)
            test_size: Proportion for test set (default 0.2 = 20%)
            random_state: Random seed for reproducibility
        """
        print("=" * 100)
        print("PREPARING DATA FOR MODEL TRAINING")
        print("=" * 100)

        # Store feature names
        self.feature_names = list(X.columns)

        # Step 1: Encode target labels
        print(f"\n[1/3] Encoding target labels...")
        self.y_encoded = self.label_encoder.fit_transform(y)
        print(f"   ✓ Encoded {len(self.label_encoder.classes_)} classes")

        # Display class mapping
        print(f"\n   Class Mapping:")
        for idx, diagnosis in enumerate(self.label_encoder.classes_):
            count = (self.y_encoded == idx).sum()
            print(f"      {idx} = {diagnosis:20s} ({count} cases)")

        # Step 2: Split data
        print(f"\n[2/3] Splitting data (train/test = {100*(1-test_size):.0f}/{100*test_size:.0f})...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, self.y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y_encoded
        )
        print(f"   ✓ Training set: {self.X_train.shape[0]} samples")
        print(f"   ✓ Test set:     {self.X_test.shape[0]} samples")
        print(f"   ✓ Features:     {self.X_train.shape[1]}")

        # Step 3: Scale features
        print(f"\n[3/3] Scaling features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"   ✓ Features scaled (mean≈0, std≈1)")

        print("\n" + "=" * 100)
        print("DATA PREPARATION COMPLETE ✅")
        print("=" * 100)

    def train_single_model(self, model_name: str, model_params: Optional[Dict] = None) -> Dict:
        """
        Train a single model.

        Args:
            model_name: Name of model ('random_forest', 'gradient_boosting', etc.)
            model_params: Optional dict of model hyperparameters

        Returns:
            Dictionary with training results
        """
        if not hasattr(self, 'X_train_scaled'):
            raise ValueError("Data not prepared. Call prepare_data() first.")

        print(f"\n{'=' * 100}")
        print(f"TRAINING: {model_name.upper().replace('_', ' ')}")
        print(f"{'=' * 100}")

        # Initialize model
        model = self._get_model(model_name, model_params)

        # Train
        print(f"\n⏳ Training {model_name}...")
        start_time = datetime.now()
        model.fit(self.X_train_scaled, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Training complete in {training_time:.2f} seconds")

        # Predictions
        y_train_pred = model.predict(self.X_train_scaled)
        y_test_pred = model.predict(self.X_test_scaled)

        # Metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        results = {
            'model_name': model_name,
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'training_time': training_time,
            'params': model_params or {},
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'timestamp': datetime.now().isoformat()
        }

        # Store model
        self.models[model_name] = results
        self.training_history.append(results)

        print(f"\n📊 Results:")
        print(f"   • Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   • Test Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   • Overfitting Gap:     {(train_accuracy - test_accuracy):.4f}")

        return results

    def train_multiple_models(self, model_configs: Optional[Dict] = None) -> pd.DataFrame:
        """
        Train multiple models and compare results.

        Args:
            model_configs: Optional dict of {model_name: params}
                          If None, uses default configurations

        Returns:
            DataFrame with comparison results
        """
        if model_configs is None:
            model_configs = self._get_default_model_configs()

        print("=" * 100)
        print(f"TRAINING {len(model_configs)} MODELS")
        print("=" * 100)

        results_list = []

        for model_name, params in model_configs.items():
            try:
                result = self.train_single_model(model_name, params)
                results_list.append({
                    'Model': model_name,
                    'Train Accuracy': result['train_accuracy'],
                    'Test Accuracy': result['test_accuracy'],
                    'Overfitting': result['train_accuracy'] - result['test_accuracy'],
                    'Training Time (s)': result['training_time']
                })
            except Exception as e:
                print(f"\n❌ Error training {model_name}: {e}")
                continue

        # Create comparison DataFrame
        df_comparison = pd.DataFrame(results_list)
        df_comparison = df_comparison.sort_values('Test Accuracy', ascending=False)
        self.comparison_results = df_comparison

        # Identify best model
        best_idx = df_comparison['Test Accuracy'].idxmax()
        self.best_model_name = df_comparison.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]['model']
        self.is_trained = True

        print("\n" + "=" * 100)
        print("MODEL COMPARISON RESULTS")
        print("=" * 100)
        print(df_comparison.to_string(index=False))
        print("\n" + "=" * 100)
        print(f"🏆 BEST MODEL: {self.best_model_name.upper().replace('_', ' ')}")
        print(f"   Test Accuracy: {df_comparison.loc[best_idx, 'Test Accuracy']:.4f} ({df_comparison.loc[best_idx, 'Test Accuracy']*100:.2f}%)")
        print("=" * 100)

        return df_comparison

    def evaluate_model(self, model_name: Optional[str] = None, verbose: bool = True) -> Dict:
        """
        Detailed evaluation of a specific model.

        Args:
            model_name: Which model to evaluate (None = best model)
            verbose: Print detailed results

        Returns:
            Dictionary with evaluation metrics
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No models trained yet.")
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")

        model_results = self.models[model_name]
        y_test_pred = model_results['y_test_pred']

        if verbose:
            print("=" * 100)
            print(f"DETAILED EVALUATION: {model_name.upper().replace('_', ' ')}")
            print("=" * 100)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_test_pred)
        precision = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        if verbose:
            print(f"\n📊 Overall Metrics:")
            print(f"   • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   • Precision: {precision:.4f}")
            print(f"   • Recall:    {recall:.4f}")
            print(f"   • F1-Score:  {f1:.4f}")

            # Classification report
            print(f"\n📋 Classification Report:")
            print("=" * 100)
            report = classification_report(
                self.y_test, 
                y_test_pred,
                target_names=self.label_encoder.classes_,
                digits=3
            )
            print(report)

            # Confusion matrix summary
            cm = confusion_matrix(self.y_test, y_test_pred)
            print(f"\n🔢 Confusion Matrix Summary:")
            print(f"   • Total predictions: {len(self.y_test)}")
            print(f"   • Correct: {np.trace(cm)} ({np.trace(cm)/len(self.y_test)*100:.1f}%)")
            print(f"   • Incorrect: {len(self.y_test) - np.trace(cm)} ({(len(self.y_test) - np.trace(cm))/len(self.y_test)*100:.1f}%)")

        return metrics

    def get_feature_importance(self, model_name: Optional[str] = None, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.

        Args:
            model_name: Which model (None = best model)
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        model = self.models[model_name]['model']

        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️ {model_name} does not have feature importance")
            return None

        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)

        print(f"\n🔝 Top {top_n} Most Important Features ({model_name}):")
        print("=" * 100)
        print(importance_df.to_string(index=False))

        return importance_df

    def predict(self, X_new: pd.DataFrame, return_proba: bool = True, 
               model_name: Optional[str] = None) -> Dict:
        """
        Make predictions on new data.

        Args:
            X_new: New feature data
            return_proba: Return probability distributions
            model_name: Which model to use (None = best model)

        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("No models trained yet. Call train_multiple_models() first.")

        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found.")
            model = self.models[model_name]['model']

        # Scale features
        X_scaled = self.scaler.transform(X_new)

        # Predict
        y_pred_encoded = model.predict(X_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)

        results = {
            'predictions': y_pred,
            'predictions_encoded': y_pred_encoded,
            'model_used': model_name
        }

        # Probabilities
        if return_proba and hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_scaled)
            results['probabilities'] = y_proba
            results['top_3_predictions'] = []

            for i in range(len(X_new)):
                top_3_idx = np.argsort(y_proba[i])[::-1][:3]
                top_3 = [(self.label_encoder.classes_[idx], y_proba[i][idx]) 
                        for idx in top_3_idx]
                results['top_3_predictions'].append(top_3)

        return results

    def cross_validate_model(self, model_name: str, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation on a model.

        Args:
            model_name: Name of model to cross-validate
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with CV results
        """
        print(f"\n{'=' * 100}")
        print(f"CROSS-VALIDATION: {model_name.upper().replace('_', ' ')} ({cv_folds} folds)")
        print(f"{'=' * 100}")

        model = self._get_model(model_name)

        # Combine train and test for CV
        X_all = np.vstack([self.X_train_scaled, self.X_test_scaled])
        y_all = np.concatenate([self.y_train, self.y_test])

        print(f"\n⏳ Running {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(model, X_all, y_all, cv=cv_folds, scoring='accuracy')

        results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max()
        }

        print(f"\n📊 Cross-Validation Results:")
        print(f"   • Mean Accuracy: {results['mean_score']:.4f} (±{results['std_score']:.4f})")
        print(f"   • Min Accuracy:  {results['min_score']:.4f}")
        print(f"   • Max Accuracy:  {results['max_score']:.4f}")
        print(f"   • All Scores:    {[f'{s:.4f}' for s in cv_scores]}")

        return results

    def tune_hyperparameters(self, model_name: str, param_grid: Dict, cv_folds: int = 3) -> Dict:
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            model_name: Name of model to tune
            param_grid: Dictionary of parameters to search
            cv_folds: Number of CV folds

        Returns:
            Dictionary with best parameters and results
        """
        print(f"\n{'=' * 100}")
        print(f"HYPERPARAMETER TUNING: {model_name.upper().replace('_', ' ')}")
        print(f"{'=' * 100}")

        model = self._get_model(model_name)

        print(f"\n⏳ Searching parameters...")
        print(f"   Parameter grid: {param_grid}")

        grid_search = GridSearchCV(
            model, param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train_scaled, self.y_train)

        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }

        print(f"\n✅ Tuning Complete!")
        print(f"\n🎯 Best Parameters:")
        for param, value in results['best_params'].items():
            print(f"   • {param}: {value}")
        print(f"\n📊 Best CV Score: {results['best_score']:.4f}")

        # Train with best params
        print(f"\n⏳ Training with best parameters...")
        self.train_single_model(model_name, results['best_params'])

        return results

    def save_model(self, filepath: str, model_name: Optional[str] = None) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Where to save the model
            model_name: Which model to save (None = best model)
        """
        if not self.is_trained:
            raise ValueError("No models trained yet.")

        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        save_data = {
            'model': self.models[model_name]['model'],
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': model_name,
            'train_accuracy': self.models[model_name]['train_accuracy'],
            'test_accuracy': self.models[model_name]['test_accuracy'],
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(save_data, filepath)
        print(f"\n💾 Model saved to: {filepath}")
        print(f"   • Model: {model_name}")
        print(f"   • Test Accuracy: {save_data['test_accuracy']:.4f}")

    def load_model(self, filepath: str) -> None:
        """
        Load a saved model from disk.

        Args:
            filepath: Path to saved model
        """
        print(f"\n📂 Loading model from: {filepath}")

        loaded_data = joblib.load(filepath)

        self.best_model = loaded_data['model']
        self.label_encoder = loaded_data['label_encoder']
        self.scaler = loaded_data['scaler']
        self.feature_names = loaded_data['feature_names']
        self.best_model_name = loaded_data['model_name']
        self.is_trained = True

        print(f"✅ Model loaded successfully!")
        print(f"   • Model: {self.best_model_name}")
        print(f"   • Test Accuracy: {loaded_data['test_accuracy']:.4f}")

    def _get_model(self, model_name: str, params: Optional[Dict] = None):
        """Initialize a model with given parameters."""
        if params is None:
            params = {}

        # Fixed: Each model gets its own correct class
        if model_name == 'random_forest':
            return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=42, **params)
        elif model_name == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000, **params)
        elif model_name == 'svm':
            return SVC(random_state=42, probability=True, **params)
        elif model_name == 'knn':
            return KNeighborsClassifier(**params)
        elif model_name == 'naive_bayes':
            return GaussianNB(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}. Available: ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm', 'knn', 'naive_bayes']")

    def _get_default_model_configs(self) -> Dict:
        """Get default configurations for multiple models."""
        return {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5
            },
            'logistic_regression': {
                'C': 1.0,
                'penalty': 'l2'
            },
            'knn': {
                'n_neighbors': 5,
                'weights': 'distance'
            },
            'naive_bayes': {}
        }

    def get_training_summary(self) -> Dict:
        """Get summary of all training."""
        if not self.training_history:
            return {'message': 'No models trained yet'}

        summary = {
            'total_models_trained': len(self.training_history),
            'best_model': self.best_model_name,
            'best_test_accuracy': self.models[self.best_model_name]['test_accuracy'],
            'all_models': list(self.models.keys()),
            'comparison_table': self.comparison_results.to_dict() if not self.comparison_results.empty else {}
        }

        return summary


