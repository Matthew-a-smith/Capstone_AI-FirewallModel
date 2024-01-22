import torch
import torch.nn as nn
import torch.optim as optim

class FirewallModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FirewallModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.encoder(x.float())  # Convert input to Float
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    # Load the processed data
    tensor_data = torch.load('dataset.pt')

    # Initialize model, criterion, and optimizer
    input_size = tensor_data.size(1)
    hidden_size = 64  # You can adjust the size of the hidden layer
    model = FirewallModel(input_size=input_size, hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(tensor_data.float())  # Convert input to Float

        # Compute the loss
        loss = criterion(outputs, tensor_data.float())  # Convert target to Float

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss after each epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Save the trained model
    model_path = 'firewall_model.pth'
    try:
        torch.save(model.state_dict(), model_path)
        print(f'Model saved successfully to {model_path}')
    except Exception as e:
        print(f'Error saving the model: {e}')
