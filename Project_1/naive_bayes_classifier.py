import pandas as pd
import math

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}  # Store P(class)
        self.feature_probs = {}  # Store P(feature|class)
        self.class_counts = {}  # To keep class counts for probability calculation
        self.feature_stats = {}  # Store statistics for continuous features
    
    def fit(self, X, y):
        # Count occurrences of each class
        self.class_counts = y.value_counts().to_dict()
        total_samples = len(y)

        # Calculate class probabilities
        self.class_probs = {cls: count / total_samples for cls, count in self.class_counts.items()}

        for cls in self.class_counts:
            self.feature_probs[cls] = {}
            self.feature_stats[cls] = {}

            # Subset data for each class
            X_class = X[y == cls]

            for col in X.columns:
                # If feature is discrete
                if X[col].dtype == 'object' or len(X[col].unique()) < 10:
                    # Calculate probabilities of discrete features
                    self.feature_probs[cls][col] = X_class[col].value_counts(normalize=True).to_dict()
                else:
                    # Store mean and variance for continuous features
                    self.feature_stats[cls][col] = (X_class[col].mean(), X_class[col].var())

    def predict_probability(self, X):
        results = []
        for idx, row in X.iterrows():
            class_scores = {}
            
            for cls in self.class_probs:
                score = math.log(self.class_probs[cls])  # Start with log(P(class))
                
                for col in X.columns:
                    value = row[col]
                    if col in self.feature_probs[cls]:  # Discrete feature
                        if value in self.feature_probs[cls][col]:
                            score += math.log(self.feature_probs[cls][col][value])
                        else:
                            score += math.log(1e-6)  # Handle unseen values with small probability
                    elif col in self.feature_stats[cls]:  # Continuous feature
                        mean, var = self.feature_stats[cls][col]
                        var = var if var > 0 else 1e-6  # Handle variance 0 case
                        score += -0.5 * math.log(2 * math.pi * var) - ((value - mean) ** 2) / (2 * var)

                class_scores[cls] = score
            
            # Choose class with highest score
            results.append(max(class_scores, key=class_scores.get))
        
        return results

    def evaluate_on_data(self, X, y_true):
        y_pred = self.predict_probability(X)
        accuracy = sum(y_pred == y_true) / len(y_true)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return y_pred
