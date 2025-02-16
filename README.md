# 🏥 Diabetes Prediction using Linear Regression

## 📌 Overview

This project builds a **Linear Regression model** to predict diabetes progression based on ten baseline variables, including age, BMI, blood pressure, and blood serum measurements. The dataset is sourced from **Scikit-Learn's Diabetes Dataset**, containing **442 instances** with **10 numerical features**.

## 📂 Dataset Information

The **Diabetes dataset** consists of:

- **Features (Independent Variables)**:
  - `age`: Age in years
  - `sex`: Gender
  - `bmi`: Body Mass Index
  - `bp`: Average Blood Pressure
  - `s1` to `s6`: Various blood serum measurements

- **Target (Dependent Variable)**:
  - A **quantitative measure** of diabetes progression **one year after baseline**.

## 📊 Project Workflow

1. **Load Data**  
   - The dataset is loaded using `sklearn.datasets.load_diabetes()`.
   - Features (`X`) and target (`y`) are extracted.

2. **Data Preprocessing**  
   - Data is **split** into **80% training** and **20% testing** using `train_test_split`.
   - Features are **scaled** using `StandardScaler()` for better model performance.

3. **Model Training**  
   - A **Linear Regression Model** is trained on the dataset using `linear_model.LinearRegression()`.

4. **Predictions & Evaluation**  
   - Predictions are made on the test dataset.
   - Model performance is evaluated using:
     - **Mean Squared Error (MSE)**
     - **R² Score (Coefficient of Determination)**

## 📈 Results

- **Model Parameters (Coefficients)**:
  ```
  [  2.69,  -9.94,  19.92,  15.19, -41.56,  
     26.39,   4.13,   6.41,  37.92,   5.31 ]
  ```

- **Intercept**: `153.45`

- **Performance Metrics**:
  - **Mean Squared Error (MSE)**: `3410.87`
  - **R² Score**: `0.49`


### 2️⃣ Regression Line Plot  
```python
plt.figure(figsize=(12, 8))
plt.scatter(y_test, Y_pred, marker='+', alpha=0.5, color='blue', label='Data Points')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2, label='Prediction Line')
plt.title("Diabetes Prediction - Linear Regression")
plt.xlabel("Actual Values (y_test)")
plt.ylabel("Predicted Values (Y_pred)")
plt.legend()
plt.grid(True)
plt.show()
```

## 📌 How to Run This Project

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/frediking/Diabetes-Prediction-Linear-Regression.git
   cd diabetes-prediction
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Python Script**  
   ```bash
   python Regression Model-Diabetes.py
   ```

4. **View Predictions & Plots**  
   The model will generate predictions, evaluation metrics, and visualizations.

## 🤝 Contribution

Contributions are welcome! Follow these steps:

1. **Fork the repository**  
2. **Create a new branch**  
   ```bash
   git checkout -b feature-name
   ```
3. **Commit your changes**  
   ```bash
   git commit -m "Added a new feature"
   ```
4. **Push to your fork**  
   ```bash
   git push origin feature-name
   ```
5. **Submit a Pull Request**

