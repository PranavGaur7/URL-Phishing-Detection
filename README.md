# URL-Phishing-Detection

A machine learning-powered tool to **detect phishing URLs** based on structural and semantic features extracted from the URL string. Easily run as a web app via Streamlit or use in your own projects for fast, accurate URL threat assessment.

***

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

***

## Overview

**URL-Phishing-Detection** leverages a comprehensive machine learning approach to identify malicious (phishing) URLs. It extracts 41 features from any given URL—including length, special characters, domain patterns, and entropy—and uses an optimally chosen Random Forest classifier to achieve high detection accuracy.

***

## Features

- **End-to-end pipeline:** Feature extraction, model training, web app, and evaluation
- **Rich feature set:** 41 URL-based attributes, including subdomain and entropy analysis
- **Multiple models evaluated:** Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, and more
- **Streamlit Web App:** Enter any URL and get an instant phishing probability diagnosis with confidence score
- **Custom Data Processing:** Flexible scripts to preprocess and label URL datasets

***

## Demo

![Project Screenshot]( Enter a URL in the web app and see its phishing risk score in real time!

***

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PranavGaur7/URL-Phishing-Detection.git
   cd URL-Phishing-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare dataset**
   - Download `file1.csv`, `file2.csv`, `phishing_dataset_features.csv` from [Dataset Folder (Google Drive)](https://drive.google.com/drive/folders/1kXXq7ZK5v9Pk3QHmV8Pc1mU2qDcSQciS?usp=sharing)

   - Place your labeled URL CSV files (e.g., `file1.csv`, `file2.csv`) in the `ProcessData/` folder.
   - Place `phishing_dataset_features.csv` in main folder
   - Run the processing script:
     ```bash
     python ProcessData/process.py
     ```

4. **Train models**
   - Open and run `train_model.ipynb` in Jupyter.
   - This will produce `best_model.pkl` and `scaler.pkl` for prediction.

5. **Launch Web App**
   ```bash
   streamlit run app.py
   ```

***

## Usage

- **Web App:** Use the Streamlit interface to check new URLs and see if they're flagged as phishing or legitimate.
- **CLI/Custom Integration:** Import feature extraction functions in your own scripts for batch URL analysis.

***

## Project Structure

```plaintext
├── app.py                  # Streamlit web app
├── feature_extraction.py   # Feature extraction logic (41 features)
├── train_model.ipynb       # Model training, evaluation & selection
├── requirements.txt        # Python dependencies
├── ProcessData/
│   └── process.py          # Data processing pipeline for raw CSVs
├── README.md               # Project documentation
```

***

## Model Details

- **Features Used:** URL length, domain statistics, special characters, digit patterns, subdomain details, entropy metrics, etc.
- **Best Model:** Random Forest (Accuracy ≈ 97.7% on validation)
- **Other Models Tested:** Logistic Regression, SVM, Decision Tree, XGBoost, SGD Classifier
- **Output:** Binary classification (Legitimate or Phishing) with a confidence score

***

***

**Developed by PranavGaur7**
