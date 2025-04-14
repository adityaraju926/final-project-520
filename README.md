# Obesity Risk Prediction

## Project Overview
This project tackles a critical public health challenge: predicting obesity risk in different demographic groups. Using machine learning, the analysis examines various demographic and socioeconomic factors to determine whether a group's obesity rate will be above or below the national median.

## What's Being Predicted
The models predict whether a specific demographic group's obesity rate is above or below the national median obesity rate. Specifically:

- **Target Variable**: Binary classification (1 or 0)
  - 1: The group's obesity rate is above the national median obesity rate
  - 0: The group's obesity rate is below or equal to the national median obesity rate

- **Prediction Context**:
  - Each prediction is made for a specific state and demographic group
  - The prediction indicates if that group's obesity rate is above the national median
  - This helps identify which groups have higher-than-median obesity rates

- **Example**:
  - If a prediction is 1 for "California, Middle-aged, High Income", it means this group's obesity rate is above the national median
  - If a prediction is 0 for "Texas, Young, Low Income", it means this group's obesity rate is at or below the national median

- **Important Note**:
  - The median is calculated across all demographic groups in the dataset
  - This provides a national benchmark for comparison
  - The prediction is not relative to state-specific medians

## Dataset Description

The dataset is sourced from the [CDC's Behavioral Risk Factor Surveillance System (BRFSS)](https://catalog.data.gov/dataset/nutrition-physical-activity-and-obesity-behavioral-risk-factor-surveillance-system), a health survey that collects data on health-related risk behaviors, chronic health conditions, and use of preventive services. The dataset contains demographic and health-related factors for different population groups across states. Key factors include:
- Age groups
- Education levels
- Income brackets
- Sex
- State-level information
- Current obesity rates

## Approach

### 1. Random Forest Model
- What it is: An ensemble of decision trees that work together to make predictions
- Key features:
  - Handles both numerical and categorical data
  - Naturally deals with missing values
  - Provides feature importance insights
  - Best parameters found (Found using GridSearchCV in train_random_forest function):
    - n_estimators: 100
    - max_depth: None
    - min_samples_leaf: 4
    - min_samples_split: 10

### 2. Neural Network Model
- What it is: A deep learning model that learns complex patterns in the data
- Key features:
  - Multiple layers for complex pattern recognition
  - Dropout for preventing overfitting
  - Adaptive learning rates
  - Best architecture found (Found in find_best_model_settings function):
    - Layer sizes: [256, 128, 64, 32]
    - Learning rate: 0.001
    - Dropout rate: 0.1
    - Epochs: 100

## Pipeline

### Data Preparation
1. Feature Engineering:
   - Created age groups:
     - 0-18: Young
     - 19-35: Young Adult
     - 36-50: Middle-aged
     - 51-65: Senior
     - 65+: Elderly
   - Defined income categories:
     - <0: No income
     - <25k: Low income
     - 25k-50k: Lower middle income
     - 50k-75k: Middle income
     - 75k-100k: Upper middle income
     - 100k: High income
   - Standardized numerical features (YearStart, Sample_Size, Age, Income)
   - Encoded categorical variables (Location, Education, Sex, Race/Ethnicity)
   - Handled missing values using median imputation
   - Selected top 10 features using SelectKBest

2. Model Training:
   - Split data: 80% training, 20% testing
   - Used 5-fold cross-validation
   - Optimized for F1 score (balancing precision and recall)
   - Stratified sampling to maintain class distribution

### Model Performance
The results show that the Random Forest model significantly outperformed the Neural Network:

Random Forest:
- Accuracy: 79.65%
- Precision: 74.31%
- Recall: 83.00%
- F1 Score: 78.42%

Neural Network:
- Accuracy: 69.01%
- Precision: 68.04%
- Recall: 57.33%
- F1 Score: 62.23%

### Key Findings
1. Random Forest excels at identifying at-risk groups (83% recall)
2. Both models show good precision in their predictions
3. The significant gap in recall (83.00% vs 57.33%) suggests Random Forest is much better at identifying true positive cases
4. Predictions are saved to two files:
   - `rf_predictions.csv`: Random Forest predictions
   - `nn_predictions.csv`: Neural Network predictions

## How to Use This Project

### Quick Start
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python main.py
```

The program will:
- Load and preprocess the data
- Train both models
- Save predictions to CSV files
- Display performance metrics

### Output Files
The program generates two prediction files:
- `rf_predictions.csv`: Random Forest predictions
- `nn_predictions.csv`: Neural Network predictions

Each file contains:
- Binary predictions (0: below average, 1: above average)
- Actual vs. predicted values
- Performance metrics

### Evaluation Metrics
Multiple metrics ensure comprehensive evaluation:
- Accuracy: Overall prediction correctness
- Precision: Minimizing false positives
- Recall: Minimizing false negatives
- F1 Score: Balance between precision and recall

### Understanding the Prediction Files

#### Random Forest Predictions (`rf_predictions.csv`)
- Format: CSV file with five columns
  - `State`: The state where the demographic group is located
  - `Actual`: The true obesity rate classification (0 or 1)
  - `Predicted`: The model's prediction (0 or 1)
- Usage:
  - Model performance analysis by state and demographic group
  - Identifying high-risk demographic groups in specific states
  - Comparing predictions with actual outcomes across different categories
  - Feature importance analysis with demographic context

#### Neural Network Predictions (`nn_predictions.csv`)
- Format: CSV file with five columns
  - `State`: The state where the demographic group is located
  - `Actual`: The true obesity rate classification (0 or 1)
  - `Predicted`: The model's prediction (0 or 1)
- Usage:
  - Comparing model performance with Random Forest across states and categories
  - Analyzing cases where models make different predictions for specific demographic groups
  - Understanding prediction patterns across states and demographic categories

Both files include a header section with:
- Total number of predictions
- Number of correct predictions
- Overall accuracy
- Additional performance metrics
- Breakdown of predictions by state and category

## Future Improvements
1. Data Enhancements:
   - Better handling of missing values
   - Additional demographic features
   - Interaction terms between features

2. Model Improvements:
   - Ensemble of both models
   - More sophisticated neural network architectures
   - Advanced feature selection methods

3. Evaluation Enhancements:
   - Time-based validation
   - Additional cross-validation folds
   - More detailed demographic analysis