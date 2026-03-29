"""
utils.py — Constants, Display Helpers, and Utility Functions
=============================================================
Contains all shared constants, song data, display formatting,
and input validation functions used across the application.

Author: [Your Name]
Course: CSA2001 — Fundamentals of AI and ML
"""

import os
import sys
import time


# ════════════════════════════════════════════════════
# FILE PATHS
# ════════════════════════════════════════════════════

DATA_FILE = "mood_music_data.csv"
MODEL_FILE = "moodtune_model.pkl"
SCALER_FILE = "moodtune_scaler.pkl"
ENCODER_FILE = "moodtune_encoder.pkl"
HISTORY_FILE = "mood_history.csv"


# ════════════════════════════════════════════════════
# MOOD & MUSIC CONSTANTS
# ════════════════════════════════════════════════════

TIME_LABELS = {
    0: "Morning",
    1: "Afternoon",
    2: "Evening",
    3: "Night"
}

GENRE_LIST = ["Pop", "Lo-fi", "Rock", "Classical", "EDM", "Acoustic"]

MOOD_FACES = {
    "Pop": "(^_^)",
    "Lo-fi": "(-_-)~",
    "Rock": "(>_<)",
    "Classical": "(u_u)",
    "EDM": "\\(^o^)/",
    "Acoustic": "(;_;)"
}

GENRE_DESCRIPTIONS = {
    "Pop": "Upbeat and feel-good vibes to match your positive energy!",
    "Lo-fi": "Chill beats to help you focus and stay in the zone.",
    "Rock": "Intense sounds to channel your energy and stress.",
    "Classical": "Peaceful melodies for your calm and relaxed state.",
    "EDM": "High-energy electronic beats for your party mood!",
    "Acoustic": "Soulful melodies that resonate with your emotions."
}

SONG_SUGGESTIONS = {
    "Pop": [
        "Blinding Lights — The Weeknd",
        "Kesariya — Arijit Singh",
        "Levitating — Dua Lipa",
        "Apna Bana Le — Arijit Singh",
        "Shape of You — Ed Sheeran",
        "Dynamite — BTS",
        "Love Story — Taylor Swift"
    ],
    "Lo-fi": [
        "Lofi Girl — Study Beats Stream",
        "Snowman — Sia (slowed + reverb)",
        "Coffee — beabadoobee",
        "Somewhere Only We Know — lo-fi remix",
        "Aesthetic — Xilo",
        "Buttercup — Jack Stauber (lo-fi)",
        "Saturn — SZA (slowed)"
    ],
    "Rock": [
        "Bohemian Rhapsody — Queen",
        "In The End — Linkin Park",
        "Thunder — Imagine Dragons",
        "Smells Like Teen Spirit — Nirvana",
        "Believer — Imagine Dragons",
        "Back in Black — AC/DC",
        "Numb — Linkin Park"
    ],
    "Classical": [
        "Moonlight Sonata — Beethoven",
        "Four Seasons (Spring) — Vivaldi",
        "Canon in D — Pachelbel",
        "Clair de Lune — Debussy",
        "Raag Yaman — Hindustani Classical",
        "River Flows in You — Yiruma",
        "Fur Elise — Beethoven"
    ],
    "EDM": [
        "Levels — Avicii",
        "Titanium — David Guetta ft. Sia",
        "Lean On — Major Lazer",
        "Animals — Martin Garrix",
        "Wake Me Up — Avicii",
        "Faded — Alan Walker",
        "Alone — Marshmello"
    ],
    "Acoustic": [
        "Let Her Go — Passenger",
        "Tum Hi Ho — Arijit Singh",
        "Perfect — Ed Sheeran",
        "Channa Mereya — Arijit Singh",
        "Photograph — Ed Sheeran",
        "Agar Tum Saath Ho — Arijit Singh",
        "Hallelujah — Leonard Cohen"
    ]
}

FEATURE_COLUMNS = [
    'energy_level', 'happiness_level', 'stress_level',
    'sadness_level', 'time_of_day', 'is_working', 'social_setting'
]

TARGET_COLUMN = 'recommended_genre'


# ════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ════════════════════════════════════════════════════

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Display MoodTune application banner."""
    print("""
    ╔══════════════════════════════════════════════╗
    ║                                              ║
    ║     ♪  M O O D T U N E  ♪                   ║
    ║     AI Music Mood Recommender                ║
    ║                                              ║
    ║     Tell me your mood,                       ║
    ║     I'll pick your music.                    ║
    ║                                              ║
    ╚══════════════════════════════════════════════╝
    """)


def print_header(title):
    """Print a formatted section header."""
    print(f"\n  {'=' * 50}")
    print(f"    {title}")
    print(f"  {'=' * 50}")


def print_line():
    """Print a separator line."""
    print(f"  {'─' * 50}")


def print_success(msg):
    """Print a success message."""
    print(f"  [OK] {msg}")


def print_error(msg):
    """Print an error message."""
    print(f"\n  [ERROR] {msg}")


def print_info(msg):
    """Print an informational message."""
    print(f"  [INFO] {msg}")


def loading(msg, seconds=1.0):
    """Display a simple loading animation."""
    for i in range(3):
        sys.stdout.write(f"\r  {msg}{'.' * (i + 1)}   ")
        sys.stdout.flush()
        time.sleep(seconds / 3)
    print(f"\r  {msg}... Done!          ")


def pause():
    """Wait for user to press Enter."""
    input("\n  Press Enter to continue...")


def mood_bar(label, value, max_val=10):
    """
    Display a visual bar for a mood value.

    Args:
        label  : Name of the mood dimension
        value  : Current value (integer)
        max_val: Maximum possible value
    """
    filled = int(value / max_val * 10)
    empty = 10 - filled
    bar = "#" * filled + "." * empty
    print(f"    {label:<12}: [{bar}] {value}/{max_val}")


def get_int_input(prompt, min_val, max_val):
    """
    Get a validated integer input from the user.

    Pressing Enter without typing returns the midpoint as default.

    Args:
        prompt  : Input prompt string
        min_val : Minimum valid value
        max_val : Maximum valid value

    Returns:
        int: Validated integer within range
    """
    while True:
        try:
            val = input(prompt).strip()
            if val == '':
                return (min_val + max_val) // 2
            val = int(val)
            if min_val <= val <= max_val:
                return val
            print(f"    [!] Enter a number between {min_val} and {max_val}")
        except ValueError:
            print(f"    [!] Please enter a valid number")