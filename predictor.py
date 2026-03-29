"""
predictor.py — Mood Prediction, Song Suggestions, and History
==============================================================
Handles all prediction-related features:
  - Mood input collection from user
  - Genre prediction using trained model
  - Song suggestion display
  - Mood tips generation
  - Prediction history tracking
  - Mood insights and pattern analysis

Author: [Your Name]
Course: CSA2001 — Fundamentals of AI and ML
"""

import os
import datetime
import numpy as np
import pandas as pd

from utils import (
    HISTORY_FILE, TIME_LABELS, SONG_SUGGESTIONS,
    GENRE_DESCRIPTIONS, MOOD_FACES,
    print_header, print_line, print_success,
    print_error, print_info, loading,
    mood_bar, get_int_input
)


class MoodPredictor:
    """
    Handles mood-based music prediction and history tracking.

    Attributes:
        model         : Trained ML model
        scaler        : Fitted StandardScaler
        label_encoder : Fitted LabelEncoder
    """

    def __init__(self, model, scaler, label_encoder):
        """
        Initialize predictor with trained components.

        Args:
            model         : Trained sklearn classifier
            scaler        : Fitted StandardScaler
            label_encoder : Fitted LabelEncoder
        """
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder

    def get_recommendation(self):
        """
        Main recommendation flow:
          1. Collect mood inputs from user
          2. Build feature array and scale
          3. Predict genre with confidence
          4. Display result with songs and tips
          5. Save to history
        """

        print_header("♪ GET MUSIC RECOMMENDATION ♪")
        print("""
  Tell me how you're feeling right now.
  Rate each on a scale of 1-10 (or press Enter for default).
        """)

        # ── Collect Mood Inputs ──
        energy = get_int_input(
            "    Energy Level    (1=tired, 10=super active)  : ", 1, 10)
        happiness = get_int_input(
            "    Happiness Level (1=unhappy, 10=very happy)  : ", 1, 10)
        stress = get_int_input(
            "    Stress Level    (1=relaxed, 10=very stressed): ", 1, 10)
        sadness = get_int_input(
            "    Sadness Level   (1=fine, 10=very sad)       : ", 1, 10)

        print(f"\n    Time of day:")
        print(f"      0 = Morning    1 = Afternoon")
        print(f"      2 = Evening    3 = Night")
        time_of_day = get_int_input(
            "    Time of Day     (0-3)                       : ", 0, 3)

        working_input = input(
            "    Are you working/studying? (y/n) [n]          : "
        ).strip().lower()
        is_working = 1 if working_input == 'y' else 0

        social_input = input(
            "    Are you with people?      (y/n) [n]          : "
        ).strip().lower()
        social = 1 if social_input == 'y' else 0

        # ── Display Mood Profile ──
        print(f"\n  YOUR MOOD PROFILE:")
        print_line()
        mood_bar("Energy", energy)
        mood_bar("Happiness", happiness)
        mood_bar("Stress", stress)
        mood_bar("Sadness", sadness)
        print(f"    {'Time':<12}: {TIME_LABELS.get(time_of_day, 'Unknown')}")
        print(f"    {'Working':<12}: {'Yes' if is_working else 'No'}")
        print(f"    {'Social':<12}: {'With People' if social else 'Alone'}")

        # ── Predict ──
        loading("Analyzing your mood")

        features = np.array([[energy, happiness, stress, sadness,
                              time_of_day, is_working, social]])
        features_scaled = self.scaler.transform(features)

        prediction_encoded = self.model.predict(features_scaled)[0]
        predicted_genre = self.label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        # Get confidence scores
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = max(probabilities) * 100
            top_indices = np.argsort(probabilities)[::-1][:3]
            top_genres = [
                (self.label_encoder.inverse_transform([i])[0],
                 probabilities[i] * 100)
                for i in top_indices
            ]
        else:
            confidence = 100.0
            top_genres = [(predicted_genre, 100.0)]

        # ── Display Result ──
        self._display_recommendation(
            predicted_genre, confidence, top_genres
        )

        # ── Mood Tips ──
        self._display_mood_tips(
            energy, happiness, stress, sadness, is_working
        )

        # ── Save History ──
        self._save_to_history(
            energy, happiness, stress, sadness, time_of_day,
            is_working, social, predicted_genre, confidence
        )

        print_success("Mood saved to history for tracking!")

    def _display_recommendation(self, genre, confidence, top_genres):
        """
        Display the genre recommendation with songs.

        Args:
            genre      : Predicted genre string
            confidence : Confidence percentage
            top_genres : List of (genre, probability) tuples
        """

        face = MOOD_FACES.get(genre, "♪")
        desc = GENRE_DESCRIPTIONS.get(genre, "")

        print(f"""
  ╔══════════════════════════════════════════════╗
  ║                                              ║
  ║  ♪  Recommended Genre: {genre:<20s} ♪  ║
  ║     Confidence: {confidence:.1f}%{' ' * (24 - len(f'{confidence:.1f}%'))}║
  ║     {face}{' ' * (39 - len(face))}║
  ║                                              ║
  ╚══════════════════════════════════════════════╝

  "{desc}"
        """)

        # Confidence breakdown
        if len(top_genres) > 1:
            print(f"  GENRE PROBABILITIES:")
            print_line()
            for g, prob in top_genres:
                bar_len = int(prob / 100 * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                marker = " << Recommended" if g == genre else ""
                print(f"    {g:<12}: [{bar}] {prob:.1f}%{marker}")

        # Song suggestions
        songs = SONG_SUGGESTIONS.get(genre, [])
        if songs:
            print(f"\n  SUGGESTED SONGS:")
            print_line()
            selected = np.random.choice(
                songs, size=min(3, len(songs)), replace=False
            )
            for i, song in enumerate(selected, 1):
                print(f"    {i}. {song}")

    def _display_mood_tips(self, energy, happiness, stress,
                           sadness, is_working):
        """
        Generate and display personalized mood tips.

        Args:
            energy, happiness, stress, sadness: Mood values (1-10)
            is_working: Whether user is working/studying
        """

        print(f"\n  MOOD TIPS:")
        print_line()

        tips_given = 0

        if stress >= 7:
            print("    > Your stress is high. Try deep breathing or a walk.")
            tips_given += 1
        if sadness >= 7:
            print("    > Feeling down? Reach out to a friend or take a break.")
            tips_given += 1
        if energy <= 3:
            print("    > Low energy? A quick stretch or coffee might help.")
            tips_given += 1
        if happiness >= 8 and energy >= 7:
            print("    > You're in a great mood! Enjoy the vibes!")
            tips_given += 1
        if is_working and stress >= 5:
            print("    > Working under stress? Take 5-min breaks every hour.")
            tips_given += 1
        if stress <= 3 and happiness >= 6:
            print("    > Feeling good! Great time for creative work.")
            tips_given += 1
        if sadness <= 3 and happiness <= 4 and stress <= 4:
            print("    > You seem neutral. Music can help lift your mood!")
            tips_given += 1

        if tips_given == 0:
            print("    > You seem balanced! Keep going. :)")

    def _save_to_history(self, energy, happiness, stress, sadness,
                         time_of_day, is_working, social, genre,
                         confidence):
        """Save the current mood entry to history CSV."""

        entry = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            'energy': energy,
            'happiness': happiness,
            'stress': stress,
            'sadness': sadness,
            'time_of_day': TIME_LABELS.get(time_of_day, "Unknown"),
            'is_working': "Yes" if is_working else "No",
            'social': "With People" if social else "Alone",
            'recommended_genre': genre,
            'confidence': round(confidence, 1)
        }

        if os.path.exists(HISTORY_FILE):
            history = pd.read_csv(HISTORY_FILE)
            history = pd.concat(
                [history, pd.DataFrame([entry])],
                ignore_index=True
            )
        else:
            history = pd.DataFrame([entry])

        history.to_csv(HISTORY_FILE, index=False)


# ════════════════════════════════════════════════════
# HISTORY & INSIGHTS (Standalone Functions)
# ════════════════════════════════════════════════════

def view_mood_history():
    """Display past mood entries and summary statistics."""

    print_header("MOOD HISTORY")

    if not os.path.exists(HISTORY_FILE):
        print("\n  No mood history yet!")
        print("  Get some recommendations first (Option 5).")
        return

    history = pd.read_csv(HISTORY_FILE)

    if len(history) == 0:
        print("\n  No entries recorded yet.")
        return

    print(f"\n  Total Mood Entries: {len(history)}\n")

    # Show recent entries
    recent = history.tail(10)

    print(f"  {'#':<3} {'Date & Time':<17} {'Energy':>6} {'Happy':>6} "
          f"{'Stress':>6} {'Genre':<12} {'Conf%':>5}")
    print(f"  {'─'*3} {'─'*17} {'─'*6} {'─'*6} {'─'*6} {'─'*12} {'─'*5}")

    for i, (_, row) in enumerate(recent.iterrows(), 1):
        print(f"  {i:<3} {row['timestamp']:<17} {row['energy']:>6} "
              f"{row['happiness']:>6} {row['stress']:>6} "
              f"{row['recommended_genre']:<12} {row['confidence']:>5}")

    # Summary
    print(f"\n  SUMMARY:")
    print_line()

    most_common = history['recommended_genre'].mode().values[0]
    avg_energy = history['energy'].mean()
    avg_happiness = history['happiness'].mean()
    avg_stress = history['stress'].mean()

    print(f"  Most Recommended Genre : {most_common}")
    print(f"  Average Energy         : {avg_energy:.1f} / 10")
    print(f"  Average Happiness      : {avg_happiness:.1f} / 10")
    print(f"  Average Stress         : {avg_stress:.1f} / 10")

    # Genre breakdown
    print(f"\n  GENRE HISTORY:")
    print_line()
    genre_counts = history['recommended_genre'].value_counts()
    for genre, count in genre_counts.items():
        pct = count / len(history) * 100
        bar = "#" * int(pct / 3)
        print(f"    {genre:<12}: {count} times ({pct:.0f}%)  {bar}")


def mood_insights():
    """Analyze mood patterns and generate personalized insights."""

    print_header("MOOD INSIGHTS & PATTERNS")

    if not os.path.exists(HISTORY_FILE):
        print("\n  Need at least a few mood entries for insights.")
        print("  Use Option 5 multiple times to build history!")
        return

    history = pd.read_csv(HISTORY_FILE)

    if len(history) < 3:
        print(f"\n  Only {len(history)} entries found. Need at least 3.")
        print("  Keep using Option 5 to build your mood profile!")
        return

    print(f"\n  Analyzing {len(history)} mood entries...\n")

    # ── Overall Profile ──
    print(f"  YOUR OVERALL MOOD PROFILE:")
    print_line()
    mood_bar("Avg Energy", round(history['energy'].mean()))
    mood_bar("Avg Happy", round(history['happiness'].mean()))
    mood_bar("Avg Stress", round(history['stress'].mean()))
    mood_bar("Avg Sadness", round(history['sadness'].mean()))

    # ── Time-of-Day Patterns ──
    if 'time_of_day' in history.columns:
        print(f"\n  TIME-OF-DAY PATTERNS:")
        print_line()
        for tod in ['Morning', 'Afternoon', 'Evening', 'Night']:
            tod_data = history[history['time_of_day'] == tod]
            if len(tod_data) > 0:
                avg_h = tod_data['happiness'].mean()
                avg_s = tod_data['stress'].mean()
                print(f"    {tod:<12}: Happy={avg_h:.1f}  "
                      f"Stress={avg_s:.1f}  ({len(tod_data)} entries)")

    # ── Personalized Insights ──
    print(f"\n  PERSONALIZED INSIGHTS:")
    print_line()

    avg_stress = history['stress'].mean()
    avg_happy = history['happiness'].mean()
    avg_energy = history['energy'].mean()
    avg_sad = history['sadness'].mean()
    most_genre = history['recommended_genre'].mode().values[0]

    if avg_stress > 6:
        print("  > You tend to be quite stressed. Try relaxation techniques.")
    elif avg_stress < 4:
        print("  > Your stress levels are generally low. That's great!")

    if avg_happy > 7:
        print("  > You're generally a happy person! Keep it up!")
    elif avg_happy < 4:
        print("  > Your happiness scores are low. Try things you enjoy.")

    if avg_energy > 7:
        print("  > You're an energetic person! Perfect for active music.")
    elif avg_energy < 4:
        print("  > Your energy tends to be low. Exercise might help.")

    if avg_sad > 5:
        print("  > Sadness levels are noticeable. Talking to someone helps.")

    print(f"\n  > Your go-to genre is: {most_genre}")
    print(f"    This means you often feel the mood associated with {most_genre}.")

    # ── Music DNA ──
    print(f"\n  YOUR MUSIC DNA:")
    print_line()
    genre_pcts = history['recommended_genre'].value_counts(normalize=True) * 100
    for genre, pct in genre_pcts.items():
        bar = "#" * int(pct / 3)
        face = MOOD_FACES.get(genre, "")
        print(f"    {genre:<12} {face:>8} : {pct:.0f}%  {bar}")