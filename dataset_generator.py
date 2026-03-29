"""
dataset_generator.py — Synthetic Mood-Music Dataset Generator
==============================================================
Creates a mood-music preference dataset based on real-world
psychological patterns of how emotions affect music choice.

Instead of downloading an external dataset, this generates
our own data using carefully designed rules with added noise.

Logic Behind Genre Mappings:
  - Happy + High Energy     → Pop, EDM
  - Sad + Low Energy        → Acoustic
  - Stressed + High Energy  → Rock
  - Calm + Studying         → Lo-fi
  - Peaceful + Relaxed      → Classical

Author: [Your Name]
Course: CSA2001 — Fundamentals of AI and ML
"""

import numpy as np
import pandas as pd
from utils import GENRE_LIST, FEATURE_COLUMNS, TARGET_COLUMN


def generate_mood_music_dataset(n_samples=500, save_path="mood_music_data.csv"):
    """
    Generate a synthetic dataset mapping mood states to music genres.

    Features:
        energy_level     (1-10) : How energetic you feel
        happiness_level  (1-10) : How happy you feel
        stress_level     (1-10) : How stressed you feel
        sadness_level    (1-10) : How sad you feel
        time_of_day      (0-3)  : Morning/Afternoon/Evening/Night
        is_working       (0/1)  : Currently working or studying?
        social_setting   (0/1)  : Alone or with people?

    Target:
        recommended_genre : Pop, Lo-fi, Rock, Classical, EDM, Acoustic

    Args:
        n_samples : Total number of entries to generate
        save_path : File path to save the CSV

    Returns:
        pandas DataFrame with the generated dataset
    """

    print("\n  [INFO] Generating mood-music dataset...")

    np.random.seed(42)

    samples_per_genre = n_samples // len(GENRE_LIST)
    remaining = n_samples - (samples_per_genre * len(GENRE_LIST))

    all_data = []

    # ─────────────────────────────────────────
    # POP — Happy, moderate-high energy, social
    # People listen to Pop when they're in a
    # generally good mood with decent energy.
    # ─────────────────────────────────────────
    for _ in range(samples_per_genre):
        all_data.append({
            'energy_level': np.random.randint(5, 9),
            'happiness_level': np.random.randint(6, 10),
            'stress_level': np.random.randint(1, 5),
            'sadness_level': np.random.randint(1, 4),
            'time_of_day': np.random.choice([0, 1, 2, 3]),
            'is_working': np.random.choice([0, 1], p=[0.7, 0.3]),
            'social_setting': np.random.choice([0, 1], p=[0.4, 0.6]),
            'recommended_genre': "Pop"
        })

    # ─────────────────────────────────────────
    # LO-FI — Calm, studying, low-medium energy
    # Students often play Lo-fi while studying
    # or working. Moderate stress but focused.
    # ─────────────────────────────────────────
    for _ in range(samples_per_genre):
        all_data.append({
            'energy_level': np.random.randint(2, 6),
            'happiness_level': np.random.randint(4, 7),
            'stress_level': np.random.randint(3, 7),
            'sadness_level': np.random.randint(2, 5),
            'time_of_day': np.random.choice([0, 1, 2, 3],
                                             p=[0.15, 0.25, 0.3, 0.3]),
            'is_working': np.random.choice([0, 1], p=[0.2, 0.8]),
            'social_setting': np.random.choice([0, 1], p=[0.8, 0.2]),
            'recommended_genre': "Lo-fi"
        })

    # ─────────────────────────────────────────
    # ROCK — Stressed, energetic, intense mood
    # People reach for Rock when they need to
    # release pent-up energy or frustration.
    # ─────────────────────────────────────────
    for _ in range(samples_per_genre):
        all_data.append({
            'energy_level': np.random.randint(6, 10),
            'happiness_level': np.random.randint(3, 7),
            'stress_level': np.random.randint(6, 10),
            'sadness_level': np.random.randint(1, 5),
            'time_of_day': np.random.choice([0, 1, 2, 3]),
            'is_working': np.random.choice([0, 1], p=[0.6, 0.4]),
            'social_setting': np.random.choice([0, 1], p=[0.5, 0.5]),
            'recommended_genre': "Rock"
        })

    # ─────────────────────────────────────────
    # CLASSICAL — Peaceful, low stress, relaxed
    # Classical music appeals during calm and
    # meditative states, especially mornings.
    # ─────────────────────────────────────────
    for _ in range(samples_per_genre):
        all_data.append({
            'energy_level': np.random.randint(2, 6),
            'happiness_level': np.random.randint(5, 9),
            'stress_level': np.random.randint(1, 4),
            'sadness_level': np.random.randint(1, 4),
            'time_of_day': np.random.choice([0, 1, 2, 3],
                                             p=[0.35, 0.2, 0.3, 0.15]),
            'is_working': np.random.choice([0, 1], p=[0.6, 0.4]),
            'social_setting': np.random.choice([0, 1], p=[0.7, 0.3]),
            'recommended_genre': "Classical"
        })

    # ─────────────────────────────────────────
    # EDM — Very high energy, happy, party mood
    # EDM is the choice for celebrations, night
    # outs, and peak energy moments.
    # ─────────────────────────────────────────
    for _ in range(samples_per_genre):
        all_data.append({
            'energy_level': min(np.random.randint(7, 11), 10),
            'happiness_level': min(np.random.randint(7, 11), 10),
            'stress_level': np.random.randint(1, 5),
            'sadness_level': np.random.randint(1, 3),
            'time_of_day': np.random.choice([0, 1, 2, 3],
                                             p=[0.05, 0.1, 0.4, 0.45]),
            'is_working': np.random.choice([0, 1], p=[0.9, 0.1]),
            'social_setting': np.random.choice([0, 1], p=[0.2, 0.8]),
            'recommended_genre': "EDM"
        })

    # ─────────────────────────────────────────
    # ACOUSTIC — Sad, emotional, low energy
    # When people feel down or nostalgic, they
    # tend to gravitate toward soulful Acoustic.
    # ─────────────────────────────────────────
    for _ in range(samples_per_genre):
        all_data.append({
            'energy_level': np.random.randint(1, 5),
            'happiness_level': np.random.randint(2, 5),
            'stress_level': np.random.randint(3, 7),
            'sadness_level': np.random.randint(6, 10),
            'time_of_day': np.random.choice([0, 1, 2, 3],
                                             p=[0.1, 0.15, 0.35, 0.4]),
            'is_working': np.random.choice([0, 1], p=[0.8, 0.2]),
            'social_setting': np.random.choice([0, 1], p=[0.85, 0.15]),
            'recommended_genre': "Acoustic"
        })

    # ── Add remaining samples to balance count ──
    for _ in range(remaining):
        genre = np.random.choice(GENRE_LIST)
        all_data.append({
            'energy_level': np.random.randint(1, 11),
            'happiness_level': np.random.randint(1, 11),
            'stress_level': np.random.randint(1, 11),
            'sadness_level': np.random.randint(1, 11),
            'time_of_day': np.random.choice([0, 1, 2, 3]),
            'is_working': np.random.choice([0, 1]),
            'social_setting': np.random.choice([0, 1]),
            'recommended_genre': genre
        })

    # ── Create DataFrame ──
    df = pd.DataFrame(all_data)

    # Clip values to valid range
    for col in ['energy_level', 'happiness_level', 'stress_level', 'sadness_level']:
        df[col] = df[col].clip(1, 10)

    # ── Add 8% noise for realism ──
    # Real human preferences have variability.
    # A happy person MIGHT still want sad songs sometimes.
    noise_count = int(len(df) * 0.08)
    noise_indices = np.random.choice(len(df), size=noise_count, replace=False)
    for idx in noise_indices:
        df.loc[idx, 'recommended_genre'] = np.random.choice(GENRE_LIST)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    df.to_csv(save_path, index=False)

    # ── Print Summary ──
    print(f"  [OK] Dataset generated successfully!")
    print(f"  {'─' * 40}")
    print(f"  Total Samples  : {len(df)}")
    print(f"  Features       : {len(df.columns) - 1}")
    print(f"  Target Classes : {len(GENRE_LIST)}")
    print(f"  Saved to       : {save_path}")
    print(f"\n  Genre Distribution:")

    for genre in GENRE_LIST:
        count = len(df[df['recommended_genre'] == genre])
        bar_len = count // 4
        bar = "#" * bar_len
        print(f"    {genre:<12}: {count:>4} samples  {bar}")

    return df


# Run standalone for testing
if __name__ == "__main__":
    df = generate_mood_music_dataset()
    print("\n  Sample rows:")
    print(df.head(10).to_string(index=False))