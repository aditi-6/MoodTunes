"""
MoodTune — AI Music Mood Recommender
======================================
Main CLI application entry point.

Handles the menu system and connects all modules:
  - dataset_generator.py  (data creation)
  - model_trainer.py      (ML pipeline)
  - predictor.py          (predictions & history)
  - utils.py              (shared utilities)

Author : Aditi Jha

Usage:
    python main.py
"""

import os
import sys
import pandas as pd

from utils import (
    DATA_FILE, clear_screen, print_banner, print_header,
    print_line, print_error, print_info, loading, pause
)
from dataset_generator import generate_mood_music_dataset
from model_trainer import MoodModelTrainer
from predictor import MoodPredictor, view_mood_history, mood_insights


# ════════════════════════════════════════════════════
# GLOBAL STATE
# ════════════════════════════════════════════════════

dataset = None
trainer = MoodModelTrainer()
predictor = None


# ════════════════════════════════════════════════════
# MENU OPTION HANDLERS
# ════════════════════════════════════════════════════

def handle_load_data():
    """Handle Option 1: Load or generate dataset."""
    global dataset

    print_header("STEP 1: LOAD / GENERATE DATASET")

    if os.path.exists(DATA_FILE):
        print(f"\n  Found existing dataset: {DATA_FILE}")
        print("  1. Use existing dataset")
        print("  2. Generate fresh dataset")

        choice = input("\n  Your choice (1/2): ").strip()

        if choice == '2':
            try:
                n = input("  Number of samples (default 500): ").strip()
                n = int(n) if n else 500
            except ValueError:
                n = 500
            dataset = generate_mood_music_dataset(
                n_samples=n, save_path=DATA_FILE
            )
        else:
            loading("Loading dataset")
            dataset = pd.read_csv(DATA_FILE)
            print(f"  [OK] Loaded {len(dataset)} mood entries")
    else:
        print("\n  No dataset found. Creating new one...")
        try:
            n = input("  Number of samples (default 500): ").strip()
            n = int(n) if n else 500
        except ValueError:
            n = 500
        dataset = generate_mood_music_dataset(
            n_samples=n, save_path=DATA_FILE
        )


def handle_explore_data():
    """Handle Option 2: Exploratory Data Analysis."""
    global dataset

    if dataset is None:
        print_error("No dataset loaded! Run Option 1 first.")
        return

    from utils import MOOD_FACES

    print_header("EXPLORATORY DATA ANALYSIS")

    # ── Basic Info ──
    print("\n  [A] DATASET OVERVIEW")
    print_line()
    print(f"  Total Entries      : {len(dataset)}")
    print(f"  Total Features     : {len(dataset.columns) - 1}")
    print(f"  Target Column      : recommended_genre")
    print(f"  Missing Values     : {dataset.isnull().sum().sum()}")
    print(f"  Duplicate Rows     : {dataset.duplicated().sum()}")

    # ── Features ──
    print("\n  [B] FEATURES USED")
    print_line()
    features = [c for c in dataset.columns if c != 'recommended_genre']
    for i, feat in enumerate(features, 1):
        col = dataset[feat]
        print(f"  {i}. {feat:<20} Range: {col.min()} - {col.max()}  "
              f"Mean: {col.mean():.1f}")

    # ── Genre Distribution ──
    print("\n  [C] GENRE DISTRIBUTION")
    print_line()
    genre_counts = dataset['recommended_genre'].value_counts()
    total = len(dataset)

    for genre, count in genre_counts.items():
        pct = count / total * 100
        bar_len = int(pct / 100 * 35)
        bar = "#" * bar_len + "." * (35 - bar_len)
        face = MOOD_FACES.get(genre, "")
        print(f"  {genre:<12} {face:>8} : [{bar}] {count} ({pct:.1f}%)")

    # ── Mood Comparison ──
    print("\n  [D] AVERAGE MOOD BY GENRE")
    print_line()
    mood_cols = ['energy_level', 'happiness_level',
                 'stress_level', 'sadness_level']

    print(f"  {'Genre':<12} | {'Energy':>7} | {'Happy':>7} | "
          f"{'Stress':>7} | {'Sad':>7}")
    print(f"  {'─'*12}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}")

    for genre in genre_counts.index:
        genre_data = dataset[dataset['recommended_genre'] == genre]
        means = [genre_data[col].mean() for col in mood_cols]
        print(f"  {genre:<12} | {means[0]:>7.1f} | {means[1]:>7.1f} | "
              f"{means[2]:>7.1f} | {means[3]:>7.1f}")

    # ── Key Patterns ──
    print("\n  [E] KEY PATTERNS OBSERVED")
    print_line()

    energy_by = dataset.groupby('recommended_genre')['energy_level'].mean()
    happy_by = dataset.groupby('recommended_genre')['happiness_level'].mean()
    stress_by = dataset.groupby('recommended_genre')['stress_level'].mean()
    sad_by = dataset.groupby('recommended_genre')['sadness_level'].mean()

    print(f"  Highest Energy Genre  : {energy_by.idxmax()} "
          f"(avg {energy_by.max():.1f})")
    print(f"  Happiest Genre        : {happy_by.idxmax()} "
          f"(avg {happy_by.max():.1f})")
    print(f"  Most Stressed Genre   : {stress_by.idxmax()} "
          f"(avg {stress_by.max():.1f})")
    print(f"  Saddest Genre         : {sad_by.idxmax()} "
          f"(avg {sad_by.max():.1f})")

    # ── Insights ──
    print("\n  [F] FEATURE INSIGHTS")
    print_line()
    print("  > EDM listeners tend to be high-energy and happy")
    print("  > Acoustic listeners tend to be sad and alone")
    print("  > Lo-fi listeners are usually studying or working")
    print("  > Rock listeners often have high stress and energy")
    print("  > Classical listeners are calm and peaceful")
    print("  > Pop listeners are generally happy, moderate energy")


def handle_train():
    """Handle Option 3: Train models."""
    global dataset, trainer, predictor

    if dataset is None:
        print_error("No dataset loaded! Run Option 1 first.")
        return

    success = trainer.train(dataset)

    if success and trainer.is_trained:
        predictor = MoodPredictor(
            trainer.best_model,
            trainer.scaler,
            trainer.label_encoder
        )


def handle_compare():
    """Handle Option 4: Compare models."""
    trainer.compare()


def handle_recommend():
    """Handle Option 5: Get music recommendation."""
    global predictor, trainer

    if predictor is None:
        # Try loading saved model
        if trainer.load_saved_models():
            predictor = MoodPredictor(
                trainer.best_model,
                trainer.scaler,
                trainer.label_encoder
            )
        else:
            print_error("No model available! Run Option 3 first.")
            return

    predictor.get_recommendation()


def handle_feature_importance():
    """Handle Option 8: Feature importance."""
    trainer.get_feature_importance()


def show_help():
    """Display help and about information."""
    print_header("HELP / ABOUT")
    print("""
  MoodTune — AI Music Mood Recommender
  ══════════════════════════════════════

  WHAT IS THIS?
  A Machine Learning system that recommends music genres
  based on your current emotional state and situation.

  HOW TO USE (Follow steps in order):
  ────────────────────────────────────
  Step 1 -> Option 1: Generate or load mood dataset
  Step 2 -> Option 2: Explore the data (see patterns)
  Step 3 -> Option 3: Train the ML models
  Step 4 -> Option 4: Compare model accuracy
  Step 5 -> Option 5: Get your music recommendation!

  MODELS USED:
  ─────────────
  1. Decision Tree    — Learns rules like "if sad AND low
                        energy, suggest Acoustic"
  2. Random Forest    — Many decision trees voting together
  3. KNN (K-Nearest   — Finds moods similar to yours in
     Neighbors)         the dataset

  KEY CONCEPTS:
  ──────────────
  * Supervised Learning (Multi-class Classification)
  * Train-Test Split (80/20 with Stratification)
  * Feature Scaling (StandardScaler)
  * Label Encoding (genres to numbers)
  * Model Evaluation (Accuracy, F1-Score)
  * Confusion Matrix Analysis
  * Feature Importance
  * Model Persistence (pickle)

  MUSIC GENRES:
  ──────────────
  Pop       (^_^)  — Happy, upbeat
  Lo-fi    (-_-)~  — Calm, focused
  Rock      (>_<)  — Intense, energetic
  Classical (u_u)  — Peaceful, relaxed
  EDM     \\(^o^)/ — Party, high energy
  Acoustic  (;_;)  — Emotional, soulful

  DEVELOPED FOR:
  ───────────────
  Course  : CSA2001 — Fundamentals of AI and ML
  Project : Bring Your Own Project (BYOP)
    """)


# ════════════════════════════════════════════════════
# MAIN MENU LOOP
# ════════════════════════════════════════════════════

def main():
    """Main application entry point and menu loop."""

    clear_screen()
    print_banner()

    while True:
        # Status indicators
        data_ok = "Y" if dataset is not None else "N"
        model_ok = "Y" if trainer.is_trained else "N"

        print(f"\n  Status: Data[{data_ok}] Model[{model_ok}]")
        print(f"""
  ╔════════════════════════════════════════════╗
  ║              MAIN MENU                     ║
  ╠════════════════════════════════════════════╣
  ║  1. Generate / Load Dataset                ║
  ║  2. Explore Dataset (EDA)                  ║
  ║  3. Train ML Models                        ║
  ║  4. Compare Models                         ║
  ║  5. ♪ Get Music Recommendation ♪           ║
  ║  6. View Mood History                      ║
  ║  7. Mood Insights & Patterns               ║
  ║  8. Feature Importance                     ║
  ║  9. Help / About                           ║
  ║  0. Exit                                   ║
  ╚════════════════════════════════════════════╝""")

        choice = input("\n  >> Enter your choice: ").strip()

        try:
            if choice == '1':
                handle_load_data()
            elif choice == '2':
                handle_explore_data()
            elif choice == '3':
                handle_train()
            elif choice == '4':
                handle_compare()
            elif choice == '5':
                handle_recommend()
            elif choice == '6':
                view_mood_history()
            elif choice == '7':
                mood_insights()
            elif choice == '8':
                handle_feature_importance()
            elif choice == '9':
                show_help()
            elif choice == '0':
                print("\n  Thanks for using MoodTune!")
                print("  Keep vibing to the right music! ♪ Goodbye!\n")
                sys.exit(0)
            else:
                print_error("Invalid choice! Enter 0-9.")

        except KeyboardInterrupt:
            print("\n")
            print_info("Interrupted. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print_error(f"Something went wrong: {str(e)}")
            print_info("Please try again.")

        pause()


# ════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
