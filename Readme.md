# 🔍 Customer Churn Prediction (Deep Learning + Streamlit App)

This is an **end-to-end machine learning project** where I built a model to predict whether a bank customer will leave (churn) or stay.  
The project uses **TensorFlow/Keras ANN** for training and **Streamlit** for creating an interactive web app for predictions.

---

## 📂 Project Workflow

1. **Data Preprocessing**
   - Loaded dataset (`Churn_Modelling.csv`)
   - Removed irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
   - Handled categorical variables using **One-Hot Encoding**
   - Scaled numerical features using **StandardScaler**

2. **Model Building**
   - Built an **Artificial Neural Network** using `TensorFlow` and `Keras`
   - Layers used:
     - Input Layer (11 neurons)
     - Hidden Layer 1 (11 neurons, Sigmoid activation)
     - Hidden Layer 2 (11 neurons, Sigmoid activation)
     - Output Layer (Sigmoid activation)
   - Loss Function: `binary_crossentropy`
   - Optimizer: `Adam`
   - Metrics: `accuracy`

3. **Model Training**
   - Train/Test Split: 80/20
   - Batch Size: 50
   - Epochs: 10
   - Achieved good accuracy on validation data

4. **Saving Model**
   - Saved trained model as `churn_model.h5`
   - Saved fitted scaler as `scaler.pkl` using `pickle`

5. **Streamlit App**
   - User-friendly web interface for real-time predictions
   - Takes customer details as input
   - Predicts churn probability and shows result with clear message:
     - ✅ **Customer likely to stay**
     - ❌ **Customer likely to leave**

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow, Keras, Matplotlib  
- **Web App:** Streamlit  
- **Other Tools:** Pickle (for saving scaler)

---


## 🚀 How to Run Locally

1. **Clone this repo**
   ```bash
   git clone https://github.com/siddhu00747/bank-customer-churn.git
   cd Customer-Churn-Prediction-Deep-Learning-Streamlit-App

