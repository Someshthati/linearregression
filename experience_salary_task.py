import pandas as pd



# house price dataset
dataset = {
    "YearsOfExperience": [2,5,1,8,3,10,4,6,7,2],
    "EducationLevel": [3,4,1,3,4,5,3,1,4,2],
    "Salary": [48000, 72000, 32000,88000,65000,115000,60000,55000,85000,40000]
}


# normalize the dataset to scalar values between 0 and 1
dataset["YearsOfExperience"] = [x / 10 for x in dataset["YearsOfExperience"]]
dataset["EducationLevel"] = [x / 5 for x in dataset["EducationLevel"]]
dataset["Salary"] = [x / 115000 for x in dataset["Salary"]] 



df = pd.DataFrame(dataset)




print(df)


# # convert pandas dataframe to numpy array
X = df[["YearsOfExperience", "EducationLevel"]].values.astype("float32")
y = df[["Salary"]].values.astype("float32")

print("Features (Years of Experience and Education Level):")
print(X)
print("\nTarget (Salary):")
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
learning_rate = 0.01

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
test_input = torch.tensor([[1.5, 0.8]])  # Years of Experience and Education Level
predicted_salary = test_input @ w + b 
print(f"\nPredicted Salary for Test Input: {predicted_salary.item() * 115000:.2f}")

