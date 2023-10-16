import numpy as np

class Loss:

    def calculate(self, output, truth):
        sample_losses = self.forward(output, truth)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss