import numpy as np
from lib.loss import Loss


class CategoricalCrossentropy(Loss):

    def forward(self, y_pred, ground_truth):
        samples = len(y_pred)
        y_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(ground_truth.shape) == 1:
            correct_confidences = y_clipped[range(samples), ground_truth]
        # Mask values - only for one-hot encoded labels
        elif len(ground_truth.shape) == 2:
            correct_confidences = np.sum(y_clipped * ground_truth, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods