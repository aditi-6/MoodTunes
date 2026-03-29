"""
model_trainer.py — ML Model Training, Evaluation, and Comparison
=================================================================
Handles the complete ML pipeline:
  1. Data preprocessing (encoding, scaling, splitting)
  2. Training three classification models
  3. Evaluating with accuracy and F1-score
  4. Comparing models and selecting the best
  5. Saving/loading trained models

Models Used:
  - Decision Tree Classifier
  - Random Forest Classifier
  - K-Nearest Neighbors Classifier

Author: [Your Name]
Course: CSA2001 — Fundamentals of AI and ML
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)

from utils import (
    MODEL_FILE, SCALER_FILE, ENCODER_FILE,
    FEATURE_COLUMNS, TARGET_COLUMN,
    print_header, print_line, print_success,
    print_error, print_info, loading
)


class MoodModelTrainer:
    """
    Handles ML model training and evaluation for MoodTune.

    Attributes:
        scaler        : Fitted StandardScaler instance
        label_encoder : Fitted LabelEncoder instance
        models        : Dict of model name -> trained model
        results       : Dict of model name -> evaluation metrics
        best_model    : Best performing model instance
        best_name     : Name of the best model
        is_trained    : Whether models have been trained
    """

    def __init__(self):
        """Initialize the trainer with empty state."""
        self.scaler = None
        self.label_encoder = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_name = None
        self.is_trained = False

    def train(self, dataset):
        """
        Run the full training pipeline.

        Steps:
          1. Separate features and target
          2. Encode labels (genre strings -> numbers)
          3. Train-test split (80/20, stratified)
          4. Feature scaling (StandardScaler)
          5. Train 3 models
          6. Evaluate and select best model
          7. Save model files

        Args:
            dataset: pandas DataFrame with mood features and genre target

        Returns:
            bool: True if training completed successfully
        """

        print_header("TRAIN ML MODELS")

        # ── Step 1: Prepare Data ──
        loading("Step 1/5: Preparing features and target")

        X = dataset[FEATURE_COLUMNS].copy()
        y = dataset[TARGET_COLUMN].copy()

        print(f"  Features : {len(FEATURE_COLUMNS)} columns")
        print(f"  Classes  : {y.nunique()} genres — {list(y.unique())}")

        # ── Step 2: Encode Labels ──
        loading("Step 2/5: Encoding genre labels")

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"  Mapping  : ", end="")
        for i, genre in enumerate(self.label_encoder.classes_):
            print(f"{genre}={i}", end="  ")
        print()

        # ── Step 3: Train-Test Split ──
        loading("Step 3/5: Splitting data (80% train, 20% test)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )

        print(f"  Training samples : {len(X_train)}")
        print(f"  Testing samples  : {len(X_test)}")

        # ── Step 4: Feature Scaling ──
        loading("Step 4/5: Scaling features (StandardScaler)")

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print_info("Features scaled (important for KNN distance calculation)")

        # ── Step 5: Train 3 Models ──
        print(f"\n  Step 5/5: Training models...\n")

        model_definitions = {
            "Decision Tree": DecisionTreeClassifier(
                random_state=42,
                max_depth=8,
                min_samples_split=5
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=5,
                metric='euclidean'
            )
        }

        self.models = {}
        self.results = {}

        for name, clf in model_definitions.items():
            # Train
            start = time.time()
            clf.fit(X_train_scaled, y_train)
            train_time = time.time() - start

            # Predict
            y_pred = clf.predict(X_test_scaled)

            # Evaluate
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Store results
            self.models[name] = clf
            self.results[name] = {
                'accuracy': acc,
                'f1_score': f1,
                'train_time': train_time,
                'y_pred': y_pred,
                'y_test': y_test
            }

            print(f"  [OK] {name:<20} | Accuracy: {acc:.4f} | "
                  f"F1: {f1:.4f} | Time: {train_time:.3f}s")

        # ── Select Best Model ──
        self.best_name = max(
            self.results,
            key=lambda x: self.results[x]['f1_score']
        )
        self.best_model = self.models[self.best_name]
        self.is_trained = True

        print(f"\n  ★ Best Model: {self.best_name}")
        print(f"    Accuracy : {self.results[self.best_name]['accuracy']:.4f}")
        print(f"    F1-Score : {self.results[self.best_name]['f1_score']:.4f}")

        # ── Save Everything ──
        self._save_models()

        return True

    def compare(self):
        """
        Display detailed comparison of all trained models.
        Shows accuracy table, F1-score bars, and confusion matrix.
        """

        if not self.is_trained:
            print_error("No models trained! Run Option 3 first.")
            return

        print_header("MODEL COMPARISON")

        # ── Comparison Table ──
        print(f"\n  {'Model':<20} | {'Accuracy':>8} | {'F1-Score':>8} | {'Time':>7}")
        print(f"  {'─'*20}-+-{'─'*8}-+-{'─'*8}-+-{'─'*7}")

        for name, res in self.results.items():
            marker = " *" if name == self.best_name else "  "
            print(f"{marker}{name:<19} | {res['accuracy']:>8.4f} | "
                  f"{res['f1_score']:>8.4f} | {res['train_time']:>6.3f}s")

        # ── Visual F1-Score Bars ──
        print(f"\n  F1-Score Comparison:")
        print_line()

        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )

        for name, res in sorted_models:
            f1 = res['f1_score']
            bar_len = int(f1 * 35)
            bar = "#" * bar_len + "." * (35 - bar_len)
            star = " << BEST" if name == self.best_name else ""
            print(f"  {name:<18} [{bar}] {f1:.4f}{star}")

        # ── Confusion Matrix for Best Model ──
        print(f"\n  Confusion Matrix ({self.best_name}):")
        print_line()

        y_test = self.results[self.best_name]['y_test']
        y_pred = self.results[self.best_name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        genres = self.label_encoder.classes_

        # Header row
        print(f"  {'':>12}", end="")
        for g in genres:
            print(f" {g[:5]:>6}", end="")
        print()

        # Data rows
        for i, genre in enumerate(genres):
            print(f"  {genre:>12}", end="")
            for j in range(len(genres)):
                print(f" {cm[i][j]:>6}", end="")
            print()

        # ── Classification Report ──
        print(f"\n  Per-Class Performance:")
        print_line()

        report = classification_report(
            y_test, y_pred,
            target_names=genres,
            output_dict=True,
            zero_division=0
        )

        print(f"  {'Genre':<12} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
        print(f"  {'─'*12}-+-{'─'*9}-+-{'─'*6}-+-{'─'*6}")

        for genre in genres:
            if genre in report:
                p = report[genre]['precision']
                r = report[genre]['recall']
                f = report[genre]['f1-score']
                print(f"  {genre:<12} | {p:>9.4f} | {r:>6.4f} | {f:>6.4f}")

        # ── Why F1-Score ──
        print(f"\n  WHY F1-SCORE?")
        print_line()
        print("  Accuracy alone can be misleading with multiple classes.")
        print("  F1-Score balances Precision (correctness) and Recall")
        print("  (completeness) — giving a more reliable overall picture.")

    def get_feature_importance(self):
        """
        Display feature importance from tree-based models.
        Shows which mood features most influence the genre prediction.
        """

        if not self.is_trained:
            print_error("No models trained! Run Option 3 first.")
            return

        print_header("FEATURE IMPORTANCE ANALYSIS")

        # Use Random Forest or Decision Tree (they have feature_importances_)
        for name in ["Random Forest", "Decision Tree"]:
            if name in self.models:
                model = self.models[name]

                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]

                    print(f"\n  Model: {name}")
                    print(f"  Feature Importance Ranking:")
                    print_line()

                    for rank, idx in enumerate(indices, 1):
                        feat = FEATURE_COLUMNS[idx]
                        imp = importances[idx]
                        bar = "#" * int(imp * 80)
                        print(f"  {rank}. {feat:<20} {bar} ({imp:.4f})")

                    # Insight
                    top_feat = FEATURE_COLUMNS[indices[0]]
                    print(f"\n  KEY INSIGHT:")
                    print_line()
                    print(f"  '{top_feat}' is the MOST important feature.")
                    print(f"  This means your {top_feat.replace('_', ' ')} has")
                    print(f"  the biggest influence on which genre suits you.")

                    return

        print_info("No tree-based model available for feature importance.")

    def _save_models(self):
        """Save model, scaler, and encoder to pickle files."""
        loading("Saving model, scaler, and encoder")

        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(self.best_model, f)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(ENCODER_FILE, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print_success(f"Model saved to {MODEL_FILE}")

    def load_saved_models(self):
        """
        Load previously saved model, scaler, and encoder.

        Returns:
            bool: True if all files loaded successfully
        """

        if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, ENCODER_FILE]):
            return False

        with open(MODEL_FILE, 'rb') as f:
            self.best_model = pickle.load(f)
        with open(SCALER_FILE, 'rb') as f:
            self.scaler = pickle.load(f)
        with open(ENCODER_FILE, 'rb') as f:
            self.label_encoder = pickle.load(f)

        self.is_trained = True
        print_success("Loaded saved model from disk")
        return True