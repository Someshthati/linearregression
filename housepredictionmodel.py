import torch

class HousePricePredictorInference:
    def __init__(self, weights_path):
        # Load the frozen tensors
        checkpoint = torch.load(weights_path)
        self.w = checkpoint['weights']
        self.b = checkpoint['bias']

    def predict(self, size, bedrooms):
        # The same neuron math from last session, but now in a class!
        input_tensor = torch.tensor([[float(size), float(bedrooms)]])
        print(f"Input Tensor: {input_tensor}")
        with torch.no_grad(): # Ensure it stays "Frozen"
            print(f"Using Weights: {self.w}, Bias: {self.b}")
            return (input_tensor @ self.w + self.b).item()

# Usage:
engine = HousePricePredictorInference("house_model_weights.pth")
print(f"Prediction: {engine.predict(1800, 4)}")