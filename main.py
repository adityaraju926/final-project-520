import torch
from preprocess import load_and_preprocess_data
from models import (
    train_random_forest, evaluate_random_forest,
    train_neural_network, evaluate_model,
    compare_models
)
import pandas as pd

def save_predictions(predictions, test_labels, test_data, filename):
    if len(predictions.shape) > 1:
        predictions = predictions.reshape(-1)
    
    # Create DataFrame with predictions, actual values, and state information
    results_df = pd.DataFrame({
        'State': test_data['State'],
        'Actual': test_labels.values,
        'Predicted': predictions
    })
    
    correct_predictions = (results_df['Actual'] == results_df['Predicted']).sum()
    total_predictions = len(results_df)
    accuracy = correct_predictions / total_predictions
    
    header = f"# Total Predictions: {total_predictions}\n# Correct Predictions: {correct_predictions}\n# Accuracy: {accuracy:.4f}\n"
    
    # Save to CSV
    with open(filename, 'w') as f:
        f.write(header)
        results_df.to_csv(f, index=False)

def main():
    train_data, test_data, train_labels, test_labels, original_states_test = load_and_preprocess_data()
    
    print("\nTraining Random Forest model")
    rf_model = train_random_forest(train_data, train_labels)
    rf_performance = evaluate_random_forest(rf_model, test_data, test_labels)
    
    # Save Random Forest predictions
    rf_predictions = rf_model.predict(test_data)
    save_predictions(rf_predictions, test_labels, original_states_test, 'rf_predictions.csv')
    print("\nPredictions have been saved to 'rf_predictions.csv'")
    
    print("\nTraining Neural Network model")
    nn_model = train_neural_network(train_data, train_labels, test_data, test_labels, train_data.shape[1])
    nn_performance = evaluate_model(nn_model, test_data, test_labels)
    
    # Save Neural Network predictions
    with torch.no_grad():
        nn_predictions = (nn_model(torch.FloatTensor(test_data)) > 0.5).float().numpy()
    save_predictions(nn_predictions, test_labels, original_states_test, 'nn_predictions.csv')
    print("\nPredictions have been saved to 'nn_predictions.csv'")
    
    # Compare performance metrics of both models
    compare_models(rf_performance, nn_performance)

if __name__ == "__main__":
    main() 