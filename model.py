import torch
import torch.nn as nn
from configs import config

class EnrollmentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnrollmentModel, self).__init__()
        
        # Define the layers
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dim)

    def forward(self, x):
        # Applying the relu function to the output of the first layer
        x = torch.relu(self.layer1(x))

        # Applying the relu function to the output of the second layer
        x = torch.relu(self.layer2(x))

        # Applying the output layer
        x = self.layer3(x)

        # Not applying the softmax because the CrossEntropy loss function applies it
        return x


def get_model_optimizer_loss(device='cpu'):
    # Creating the model
    model = EnrollmentModel(input_dim=config['input_dim'], output_dim=config['output_dim'])
    model = model.to(device)
    
    # Defining the loss function
    loss_function = torch.nn.CrossEntropyLoss()
    
    # Defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    return model, optimizer, loss_function




        