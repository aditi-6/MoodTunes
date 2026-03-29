# 🎵 MoodTune — AI Music Mood Recommender

**MoodTune** is a complete, interactive Command-Line Interface (CLI) Machine Learning application developed as the Capstone/BYOP project for the **Fundamentals of AI and ML (CSA2001)** course.

It bridges the gap between human psychology and music by using Classification algorithms to predict the perfect music genre based on a user's current emotional state and environmental context.


## Project Overview

Research indicates that people instinctively select music to regulate or match their emotions. When energetic and happy, we gravitate toward Pop or EDM; when stressed but active, we might prefer Rock; and when studying, Lo-fi helps us focus.

**The Problem:** Manually figuring out what to listen to when you are experiencing a complex mix of emotions (e.g., high stress, low energy, late at night) can be tedious.
**The Solution:** MoodTune takes 7 contextual and emotional inputs from the user, processes them through trained Machine Learning models, and outputs a highly accurate genre recommendation along with specific song suggestions and personalized mood insights.

Instead of relying on pre-existing datasets, this project dynamically generates a synthetic, rule-based dataset with controlled statistical noise to mimic real human unpredictability.

---

## Key Features

* **Dynamic Data Generation:** Generates a 300+ sample synthetic dataset mapped to 6 music genres (Pop, Lo-fi, Rock, Classical, EDM, Acoustic).
* **Exploratory Data Analysis (EDA):** Built-in module to view statistical distributions, feature correlations, and mood-to-genre averages.
* **Multi-Model Training Engine:** Simultaneously trains, evaluates, and compares **Decision Tree**, **Random Forest**, and **K-Nearest Neighbors (KNN)** classifiers.
* **Intelligent Preprocessing:** Implements `StandardScaler` for distance-based algorithms and `LabelEncoder` for categorical targets.
* **Interactive Prediction System:** A user-friendly CLI questionnaire that captures mood and returns predictions with **confidence probability scores**.
* **Model Persistence:** Uses `pickle` to save the best-performing model, scaler, and encoder to disk so they don't need to be retrained every time.
* **History & Analytics:** Tracks user predictions in a CSV file and provides personalized psychological insights based on historical mood trends.

---

## ⚙️ Machine Learning Pipeline

The project demonstrates a complete end-to-end ML lifecycle:

1. **Feature Engineering:** 7 features are used: `energy_level`, `happiness_level`, `stress_level`, `sadness_level` (all 1-10 scales), plus `time_of_day`, `is_working`, and `social_setting`.
2. **Data Splitting:** Data is split 80/20 using `train_test_split` with `stratify=y` to ensure balanced genre representation in both sets.
3. **Scaling:** Features are normalized using `StandardScaler` to ensure the KNN algorithm calculates Euclidean distances accurately without feature dominance.
4. **Evaluation Metric:** **F1-Score (Weighted)** is used as the primary evaluation metric instead of standard accuracy, ensuring the model performs well across all genre classes without bias.
5. **Feature Importance:** Utilizes the Random Forest model to extract and display which mood indicators carry the most weight in determining musical taste.

---

## Repository Structure

The project follows a clean, modular Object-Oriented Programming (OOP) architecture, split across 5 distinct Python files to ensure maintainability:

```text
MoodTune/
├── main.py                 # Application entry point & CLI Menu Controller
├── utils.py                # Shared constants, UI formatters, and helpers
├── dataset_generator.py    # Synthetic dataset creation and logic
├── model_trainer.py        # ML preprocessing, training, and evaluation
├── predictor.py            # Inference engine, history tracking, and insights
├── requirements.txt        # Python package dependencies
└── README.md               # Detailed project documentation
```
---


## Prerequisites & Environment Setup

To run this project, ensure you have the following installed on your system:

1. Python: Version 3.8 or higher. (Verify by running python --version in terminal).
2. Git: To clone the repository.
3. Terminal/Command Prompt: The application runs entirely within the CLI.

---


## Installation Guide
Please follow these exact steps to set up the project on your local machine.

Step 1: Clone the Repository

Bash

```
git clone https://github.com/YOUR_GITHUB_USERNAME/MoodTune.git
cd MoodTune
```
Step 2: Create a Virtual Environment (Highly Recommended)
Creating a virtual environment ensures that the project's dependencies do not interfere with your global Python installation.

For Windows:

Bash
```
python -m venv venv
venv\Scripts\activate
```
For macOS/Linux:

Bash
```
python3 -m venv venv
source venv/bin/activate
```
Step 3: Install Required Dependencies

Bash
```
pip install -r requirements.txt
```

---


## Step-by-Step Execution Guide

To start the application, run the following command in your terminal:

Bash
```
python main.py
```
Once the CLI launches, you must follow the steps in this specific order for the first run:

-Select Option [1] - Generate / Load Dataset:
  Choose sub-option 2 to generate a fresh dataset.
  Press Enter to accept the default sample size (or type a number > 100).
-Select Option [2] - Explore Dataset (Optional but recommended):
  View the statistical breakdown and feature distributions of the generated data.
-Select Option [3] - Train ML Models:
  The system will automatically scale the data, train 3 models, select the best one based on F1-Score, and save it to disk.
-Select Option [4] - Compare Models:
  View a detailed comparative table and Confusion Matrix of the models.
-Select Option [5] - ♪ Get Music Recommendation ♪:
  This is the core feature. Answer the 7 prompts about your current mood. The system will predict your ideal music genre, display a confidence score, and suggest specific songs!
-Select Option [6] & [7]:
  Use these after making a few predictions to view your mood history and get personalized psychological insights.

---


##Troubleshooting

1. ModuleNotFoundError: No module named 'pandas'

  Fix: You haven't installed the dependencies. Ensure you run pip install -r requirements.txt.

2. [ERROR] No models trained! Run Option 3 first.

  Fix: You are trying to predict or compare models before training them. The pipeline requires Data Generation (Opt 1) -> Training (Opt 3) -> Prediction (Opt 5).
  
3. The least populated class in y has only X members...

  Fix: This happens if you generated a dataset with too few samples (e.g., 5 samples). Delete the mood_music_data.csv file, restart the app, and let it generate the default 300+ samples.
