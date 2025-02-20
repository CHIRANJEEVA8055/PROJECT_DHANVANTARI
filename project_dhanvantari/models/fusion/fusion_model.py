import numpy as np


def weighted_ensemble(cnn_preds, nlp_preds, cnn_weight=0.7, nlp_weight=0.3):
    """Combines CNN and NLP model predictions using weighted ensembling."""
    assert len(cnn_preds) == len(nlp_preds), "Prediction arrays must have the same length"

    # Normalize weights to sum to 1
    total_weight = cnn_weight + nlp_weight
    cnn_weight /= total_weight
    nlp_weight /= total_weight

    # Weighted sum of predictions
    fused_predictions = (cnn_weight * np.array(cnn_preds)) + (nlp_weight * np.array(nlp_preds))
    return fused_predictions


if __name__ == "__main__":
    # Example test data (Random dummy predictions for debugging)
    cnn_preds = [0.8, 0.2, 0.6, 0.9]
    nlp_preds = [0.7, 0.3, 0.5, 0.8]

    fused_results = weighted_ensemble(cnn_preds, nlp_preds)
    print("Fused Predictions:", fused_results)
