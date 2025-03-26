from fastapi import FastAPI
import torch
import uvicorn
import numpy as np
from pydantic import BaseModel
from model import EnrollmentModel  # Import your model
from configs import config  # Import configuration
from data import preprocess
# Initialize FastAPI app
app = FastAPI()

# Load trained model
device = torch.device("cpu")
model = EnrollmentModel(input_dim=config["input_dim"], output_dim=config["output_dim"])
model.load_state_dict(torch.load(config["model_save_path"], map_location=device))
model.eval()

# Define input schema
class InputData(BaseModel):
    age: float
    salary: float
    tenure_years: float
    gender: str
    marital_status: str
    employment_type: str
    region: str
    has_dependents: str

# Encoding function
def encode_input(data: InputData):
    encoded = {
        "age": data.age,
        "salary": data.salary,
        "tenure_years": data.tenure_years,
        "gender": data.gender,
        "marital_status": data.marital_status,
        "employment_type": data.employment_type,
        "region": data.region,
        "has_dependents": data.has_dependents,
    }
    encoded = preprocess(encoded)
    encoded = torch.tensor(encoded, dtype=torch.float).unsqueeze(0)
    return encoded

# API endpoint for prediction
@app.post("/predict/")
def predict(data: InputData):
    input_tensor = encode_input(data)
    with torch.no_grad():
        print(input_tensor)
        output = model(input_tensor)
        print(output)
        prediction = torch.argmax(output, dim=1).item()
    return {"enrollment_prediction": prediction}

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
