import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Scale the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Train the neural network
for epoch in range(1000):
    running_loss = 0.0
    optimizer.zero_grad()

    # Forward pass
    outputs = net(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long))

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    # Print the current loss
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, loss: {running_loss:.3f}")
        running_loss = 0.0

# Evaluate the trained model on the test set
outputs = net(torch.tensor(X_test, dtype=torch.float32))
_, predicted = torch.max(outputs.data, 1)
accuracy = (predicted == torch.tensor(y_test)).sum().item() / len(y_test)
print(f"Test accuracy: {accuracy:.3f}")
