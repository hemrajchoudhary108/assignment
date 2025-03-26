Here’s the properly formatted version of your content:  

---

# **Assignment**  

## **Data Preprocessing**  
A key observation in the dataset is its simplicity in terms of complexity. It contains only three continuous columns—**age, income, and tenure**—while all other columns are categorical. This makes the dataset relatively straightforward, allowing for the use of a **smaller, simpler model** for classification.  

## **Feature Transformation**  

### **1. Age Normalization**  
The age feature is standardized using the mean and standard deviation to ensure better and faster learning:  
```python
age = (age - age_mean) / age_std
```

### **2. Gender Encoding**  
Since gender is a categorical variable, it is encoded as follows to maintain symmetry in the feature space:  
- **Female** → **-0.5**  
- **Male** → **0.5**  

### **3. Marital Status Encoding**  
The marital status categories are mapped to numerical values to create an evenly distributed space:  
- **Single** → **-0.6**  
- **Divorced** → **-0.2**  
- **Married** → **0.2**  
- **Widowed** → **0.6**  

### **4. Salary Normalization**  
The salary feature is standardized using its mean and standard deviation to improve model efficiency and convergence speed:  
```python
salary = (salary - mean_salary) / std_salary
```

### **5. Employment Type Encoding**  
Employment type is represented numerically as follows:  
- **Part-Time** → **-0.5**  
- **Full-Time** → **0**  
- **Contract** → **0.5**  

### **6. Region Encoding**  
The geographic regions are categorized using a similar encoding strategy as marital status:  
- **West** → **-0.6**  
- **Midwest** → **-0.2**  
- **Northeast** → **0.2**  
- **South** → **0.6**  

### **7. Dependent Status Encoding**  
The presence of dependents is converted into numerical values:  
- **Yes** → **-0.5**  
- **No** → **0.5**  

### **8. Tenure Normalization**  
The tenure in years is standardized using the mean and standard deviation, helping the model learn more effectively:  
```python
tenure_years = (tenure_years - mean_tenure_years) / std_tenure_years
```

---

## **Model Development**  
Since this is a **binary classification problem**, a **simple neural network** with two hidden layers was chosen. Additional layers were avoided to prevent **overfitting**, as the dataset is relatively small and lacks significant complexity.  

---

## **Final Evaluation Results**  

✅ **Train Accuracy:** **98.26%**  
✅ **Test Accuracy:** **98.12%**  

---

## **Instructions to Run the Code**  

```bash
# 1. Clone the repository
git clone https://github.com/hemrajchoudhary108/assignment.git

# 2. Navigate to the repository directory
cd assignment

# 3. Install dependencies from requirements.txt
pip install -r requirements.txt

# 4. Train the model
python train.py
```

---

## **Key Takeaways and Next Steps**  

1. The training and test accuracies are **closely aligned**, indicating that the model generalizes well instead of overfitting.  
2. Since the accuracy is **not at 100%**, there is still room for improvement. Potential enhancements include:  
   - **Implementing Batch Normalization** to stabilize training.  
   - **Adding more layers** while reducing the size of each layer for better feature extraction.  
   - **Experimenting with different batch sizes** and learning rates.  
   - **Using adaptive learning rate techniques** such as **learning rate decay** to optimize convergence.  

---

## Running the FastAPI and Streamlit Applications  

This project includes two components:  

1. **FastAPI Server** – Serves the trained model and provides predictions via an API.  
2. **Streamlit UI** – A user-friendly interface for inputting data and obtaining predictions.  

### **Note:** Ensure the model is trained before running the applications. Use `python train.py` to train the model if not already done.  

### **Step 1: Start the FastAPI Server**  

Run the following command to start the FastAPI backend:  

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

- The API will be accessible at: **`http://localhost:8001`**  
- To check the interactive API docs, open: **`http://localhost:8001/docs`**  

### **Step 2: Start the Streamlit App**  

In a new terminal window, run:  

```bash
streamlit run app_streamlit.py --server.port 8501
```

- The Streamlit UI will be available at: **`http://localhost:8501`**  
- Use the form to input data and get predictions from the FastAPI server.  
