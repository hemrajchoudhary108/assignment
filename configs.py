# Configuration file for the project

config = {
    # Normalization parameters for 'age'
    'age': {
        'mean': 43.002000,  # Mean age in the dataset
        'std': 12.285800    # Standard deviation of age
    },
    
    # Encoding for 'gender' (centered around 0 for symmetry)
    'gender': {
        'Female': -0.5,
        'Male': 0.5,
        'Other': 0.0  # Neutral value for "Other"
    },

    # Encoding for 'marital_status' (spaced evenly in [-1, 1] range)
    'marital_status': {
        'Single': -0.6,
        'Divorced': -0.2,
        'Married': 0.2,
        'Widowed': 0.6
    },

    # Normalization parameters for 'salary'
    'salary': {
        'mean': 65032.967907,  # Mean salary
        'std': 14923.958446    # Standard deviation of salary
    },

    # Encoding for 'employment_type'
    'employment_type': {  
        'Part-time': -0.5,
        'Full-time': 0.0,
        'Contract': 0.5
    },

    # Encoding for 'region' (spaced evenly in [-1, 1] range)
    'region': {
        'West': -0.6,
        'Midwest': -0.2,
        'Northeast': 0.2,
        'South': 0.6
    },

    # Encoding for 'has_dependents'
    'has_dependents': {
        'No': 0.5,
        'Yes': -0.5
    },

    # Normalization parameters for 'tenure_years'
    'tenure_years': {
        'mean': 3.967720,  # Mean years of tenure
        'std': 3.895488    # Standard deviation of tenure
    },

    # Order of columns to be used for model input
    'column_order': [
        'age', 'gender', 'marital_status', 'salary',
        'employment_type', 'region', 'has_dependents', 'tenure_years'
    ],

    # Training hyperparameters
    'epochs': 10,             # Number of training epochs
    'batch_size': 32,         # Batch size for training
    'learning_rate': 0.001,   # Learning rate for the optimizer

    # Model architecture parameters
    'input_dim': 8,           # Number of input features
    'output_dim': 2,          # Number of output classes

    # Path to save the trained model
    'model_save_path': 'model.pth'
}
