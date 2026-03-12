import pandas as pd



# house price dataset
dataset = {
    "Size": [1000, 1500, 2000, 2500, 3000],
    "Price": [200000, 300000, 400000, 500000, 600000],
    "Bedrooms": [2, 3, 4, 5, 6]
}


df = pd.DataFrame(dataset)


print(df)


# # convert pandas dataframe to numpy array
X = df[["Size", "Bedrooms"]].values.astype("float32")
y = df[["Price"]].values.astype("float32")

print("Features (Size and Bedrooms):")
print(X)
print("\nTarget (Price):")
print(y)


# Convert NumPy arrays to PyTorch tensors
import torch    
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)




# now update tensors to require gradients
X_tensor.requires_grad_(True)
y_tensor.requires_grad_(True)


# Initialize weights and bias
w = torch.randn(2, 1, requires_grad=True)  # 2 features
b = torch.randn(1, requires_grad=True)     # 1 bias
# print("\nInitial Weights:")
# print(w)
# print("\nInitial Bias:")
# print(b)

# Training loop
learning_rate = 0.0000001

for i in range(100):
    # Forward pass: Compute predicted y
    y_pred = X_tensor @ w + b  # Matrix multiplication


    # print(f"Iteration {i}: Predicted Prices:")
    # print(y_pred)

    # Compute and print loss
    loss = torch.mean((y_pred - y_tensor) ** 2)
    
    # Backward pass: compute gradient of the loss with respect to all tensors with requires_grad=True
    loss.backward()

    # Update weights and bias using gradient descent
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Zero gradients after updating
        w.grad.zero_()
        b.grad.zero_()

    if i % 20 == 0:
        print(f"Iteration {i}: Loss = {loss.item():.2f}")

torch.save({'weights': w, 'bias': b}, "house_model_weights.pth")

# # Final test
# test_input = torch.tensor([[1800.0, 4.0]])  # Size and Bedrooms
# predicted_price = test_input @ w + b
# print(f"\nPredicted Price for Test Input: {predicted_price.item():.2f}")

