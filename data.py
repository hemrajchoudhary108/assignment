import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from configs import config

class MyDataset(Dataset):
    def __init__(self, path=None):
        """Custom dataset class for loading data from CSV files."""
        self.path = path
        self.df = pd.read_csv(self.path)  # Load dataset into a DataFrame
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index):
        """Fetches a single data sample (features and label)."""
        item = self.df.iloc[index].to_dict()
        x = preprocess(item)  # Preprocess the features
        x = torch.tensor(x)  # Convert to a tensor
        y = self.df.iloc[index]['enrolled']  # Extract label
        y = torch.tensor(y)  # Convert label to tensor
        return x, y
    
def preprocess(item):
    """Preprocesses a single row of data by normalizing numeric features 
    and mapping categorical features to numerical values."""
    # item is a dictionary with keys as column names and values as the corresponding data.
    array = []
    
    for column_name in config['column_order']:
        value = item[column_name]
        
        # Normalize numerical features
        if column_name in ['age', 'salary', 'tenure_years']:
            value = (value - config[column_name]['mean']) / config[column_name]['std']
        else:
            value = config[column_name][value]  # Map categorical values
        
        array.append(value)
    
    return array

def get_data_loader(path=None):
    """Creates a DataLoader from the dataset."""
    if path is None:
        raise ValueError("Please provide the path to the dataset")

    dataset = MyDataset(path=path)  # Load dataset
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)  # Create DataLoader
    return data_loader
