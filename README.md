# Movie Recommender System & A/B Testing Framework

A production-grade movie recommendation system demonstrating Collaborative Filtering (SVD), Popularity baselines, and a rigorous A/B Testing experimentation framework.

## Features
- **Data Pipeline**: Automated download an processing of MovieLens 100k data.
- **Recommendations**: 
  - **Popularity Baseline**: Fallback for cold-start users.
  - **SVD (Matrix Factorization)**: Collaborative filtering for personalized suggestions.
- **A/B Testing Simulator**: 
  - Deterministic user bucketing (Hashing).
  - Simulation engine using held-out test data ("Future Ground Truth") to model user behavior.
  - Statistical analysis (Z-Test) to measure lift.
- **Interactive Dashboard**: Streamlit app for exploring recommendations and running live experiment simulations.

## Tech Stack
- **Python 3.11** (Pyenv + Pip)
- **ML**: Scikit-Learn (TruncatedSVD), NumPy, Pandas
- **Stats**: SciPy, Statsmodels
- **App**: Streamlit, Plotly

## Installation

1. **Clone the repository**
2. **Setup Python Environment** (Recommended: `pyenv`)
   Ensure you have Python 3.11+ installed.
   ```bash
   pyenv local 3.11.14  # or your preferred version
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Data Setup
Download and process the MovieLens dataset:
```bash
python src/data/loader.py
```

### 2. Run the Dashboard
Launch the interactive application:
```bash
streamlit run src/app/dashboard.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

- **User View**: Login as any user ID (e.g., 42) to see your assigned group and recommendations.
- **Admin View**: Go to 'Admin Dashboard' to run a full A/B test simulation and see the statistical results.

## Docker Support
Build and run the application in a container:
```bash
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender
```

## Development & Testing
Run the test suite to verify model mechanics and experiment logic:
```bash
pip install pytest
python -m pytest tests/
```

## Running the Experiment Script
You can also run the A/B test simulation directly from the CLI:
```bash
export PYTHONPATH=$PYTHONPATH:.
python src/experiment/ab_test.py
```

## Project Structure
```
├── data/                   # Raw and processed data
├── notebooks/              # EDA and Prototyping
├── src/
│   ├── app/                # Streamlit Dashboard
│   ├── data/               # Data Loaders
│   ├── experiment/         # A/B Testing Logic & Stats
│   └── models/             # Recommender Classes
└── requirements.txt        # Dependencies
```
