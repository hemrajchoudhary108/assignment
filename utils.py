import torch

def get_device():
    """Returns the best available device: CUDA (GPU), MPS (Apple), or CPU."""
    device = 'cpu'  # Default to CPU
    if torch.cuda.is_available():
        device = 'cuda'  # Use CUDA if available
    if torch.backends.mps.is_available():
        device = 'mps'  # Use MPS (Apple Metal) if available
    return device

def evaluate_accuracy(device, model, dataloader):
    """Evaluates the model's accuracy on a given dataset."""
    accuracies = []
    
    for x, y in dataloader:
        x = x.to(device)  # Move inputs to the selected device
        y = y.to(device)  # Move labels to the selected device
        
        with torch.no_grad():  # Disable gradient computation for evaluation
            y_pred = model(x.float())  # Get model predictions
            y_pred = torch.argmax(y_pred, dim=1)  # Get predicted class index
            total = torch.sum(y == y_pred).item()  # Count correct predictions
            accuracy = total / y.shape[0]  # Compute batch accuracy
            accuracies.append(accuracy)
    
    return sum(accuracies) / len(accuracies)  # Compute overall accuracy
