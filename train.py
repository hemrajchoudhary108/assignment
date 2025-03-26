from webbrowser import get
import torch
from configs import config
from data import get_data_loader
import os
from model import get_model_optimizer_loss
from utils import evaluate_accuracy, get_device

def main(device='cpu'):
    # Initialize model, optimizer, and loss function
    model, optimizer, loss_function = get_model_optimizer_loss(device=device)

    # Define data paths
    train_path = os.path.join(os.getcwd(), "train.csv")
    test_path = os.path.join(os.getcwd(), "test.csv")

    # Load data
    train_dataloader = get_data_loader(train_path)
    test_dataloader = get_data_loader(test_path)

    losses = []

    for epoch in range(config['epochs']):
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)  # Move data to device

            optimizer.zero_grad()  # Zero the gradients
            y_pred = model(x.float())  # Forward pass
            loss = loss_function(y_pred, y)  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            losses.append(loss.item())  # Track loss
        
        # Evaluate model
        train_accuracy = evaluate_accuracy(device, model, train_dataloader)
        test_accuracy = evaluate_accuracy(device, model, test_dataloader)
        print(f"Epoch {epoch+1}: Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}")

    # Save trained model
    torch.save(model.state_dict(), config['model_save_path'])

if __name__ == '__main__':
    device = get_device()  # Get the device (CPU/GPU)
    main(device=device)
