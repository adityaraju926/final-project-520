import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

class ObesityPredictor(nn.Module):
    def __init__(self, num_features, layer_sizes, dropout_chance=0.2):
        super(ObesityPredictor, self).__init__()
        self.layers = nn.ModuleList()
        
        current_size = num_features
        for next_size in layer_sizes:
            self.layers.append(nn.Linear(current_size, next_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_chance))
            current_size = next_size
        
        self.layers.append(nn.Linear(current_size, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features):
        for layer in self.layers:
            features = layer(features)
        return self.sigmoid(features)

def train_model_with_settings(train_data, train_labels, val_data, val_labels, num_features, 
                            layer_sizes, learning_speed, dropout_chance, 
                            num_epochs=100):
    # Prepare the data for training by converting to tensors
    train_features = torch.FloatTensor(train_data)
    train_targets = torch.FloatTensor(train_labels.values).reshape(-1, 1)
    val_features = torch.FloatTensor(val_data)
    val_targets = torch.FloatTensor(val_labels.values).reshape(-1, 1)
    
    # Set up the model and training tools
    model = ObesityPredictor(num_features, layer_sizes, dropout_chance)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_speed)
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(train_features)
        loss = loss_function(predictions, train_targets)
        loss.backward()
        optimizer.step()
        
        # Check status every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_predictions = model(val_features)
                val_loss = loss_function(val_predictions, val_targets)
                print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    
    return model

def find_best_model_settings(train_data, train_labels, val_data, val_labels, num_features):
    print("Searching for the best model settings")
    
    # Define potential params
    possible_settings = {
        'layer_sizes': [
            [64, 32],
            [128, 64, 32],
            [256, 128, 64, 32]
        ],
        'learning_speed': [0.001, 0.0005, 0.0001],
        'dropout_chance': [0.1, 0.2, 0.3],
        'num_epochs': [50, 100]
    }
    
    best_loss = float('inf')
    best_settings = None
    
    # Create validation data
    val_features = torch.FloatTensor(val_data)
    val_targets = torch.FloatTensor(val_labels.values).reshape(-1, 1)
    
    # Trying different combinations of params
    for layer_sizes in possible_settings['layer_sizes']:
        for learning_speed in possible_settings['learning_speed']:
            for dropout_chance in possible_settings['dropout_chance']:
                for num_epochs in possible_settings['num_epochs']:
                    print(f"\nTrying: layers={layer_sizes}, learning={learning_speed}, dropout={dropout_chance}, epochs={num_epochs}")
                    
                    # Train model with current params
                    model = train_model_with_settings(
                        train_data, train_labels, val_data, val_labels,
                        num_features, layer_sizes, learning_speed, dropout_chance,
                        num_epochs
                    )
                    
                    # Evaluating performance
                    model.eval()
                    with torch.no_grad():
                        val_predictions = model(val_features)
                        val_loss = nn.BCELoss()(val_predictions, val_targets)
                        
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_settings = {
                                'layer_sizes': layer_sizes,
                                'learning_speed': learning_speed,
                                'dropout_chance': dropout_chance,
                                'num_epochs': num_epochs
                            }
    
    print("\nFound best settings:")
    print(best_settings)
    print(f"Best validation loss: {best_loss:.4f}")
    
    return best_settings

def train_neural_network(train_data, train_labels, test_data, test_labels, num_features):
    # Split training data
    from sklearn.model_selection import train_test_split
    train_split, val_data, train_labels_split, val_labels = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42
    )
    
    best_settings = find_best_model_settings(
        train_split, train_labels_split, val_data, val_labels, num_features
    )
    
    final_model = train_model_with_settings(
        train_data, train_labels, test_data, test_labels,
        num_features,
        best_settings['layer_sizes'],
        best_settings['learning_speed'],
        best_settings['dropout_chance'],
        best_settings['num_epochs']
    )
    
    return final_model

def evaluate_model(model, test_data, test_labels):
    test_features = torch.FloatTensor(test_data)
    
    model.eval()
    with torch.no_grad():
        predictions = model(test_features)
        binary_predictions = (predictions > 0.5).float()
        
        performance = {
            'accuracy': accuracy_score(test_labels, binary_predictions),
            'precision': precision_score(test_labels, binary_predictions),
            'recall': recall_score(test_labels, binary_predictions),
            'f1': f1_score(test_labels, binary_predictions)
        }
    
    return performance

def train_random_forest(train_data, train_labels):    
    possible_settings = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model_tuner = GridSearchCV(
        estimator=base_model,
        param_grid=possible_settings,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    model_tuner.fit(train_data, train_labels)
    print(f"Best params found: {model_tuner.best_params_}")
    final_model = model_tuner.best_estimator_
    
    return final_model

def evaluate_random_forest(model, test_data, test_labels):
    predictions = model.predict(test_data)
    
    performance = {
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions),
        'recall': recall_score(test_labels, predictions),
        'f1': f1_score(test_labels, predictions)
    }
    
    return performance

def compare_models(rf_performance, nn_performance):
    print("\nModel Comparison:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Random Forest':<15} {'Neural Network':<15}")
    print("-" * 50)
    
    for metric in rf_performance.keys():
        print(f"{metric:<15} {rf_performance[metric]:.4f}        {nn_performance[metric]:.4f}")
    
    print("-" * 50) 