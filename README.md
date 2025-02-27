# 🏥 Diabetes Prediction using Linear & Ridge Regression

## 📌 Overview

This project builds a **Regression model** to predict diabetes progression based on ten baseline variables, including age, BMI, blood pressure, and blood serum measurements. The dataset is sourced from **Scikit-Learn's Diabetes Dataset**, containing **442 instances** with **10 numerical features**.

The project now includes:
- **Linear Regression**
- **Ridge Regression** (L2 Regularization)
- **Feature Scaling** using `StandardScaler`
- **Performance Comparison (Before vs. After Ridge Regression)**


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
   - Data is **split** into **80% training** and **20% testing** using `train_test_split()`.
   - Features are **scaled** using `StandardScaler` to improve model performance.(Linear Regression)
   - **Polynomial Features** are created     using scaled feature dataset X     (Ridge Regression)
     
3. **Model Training**  
   - **Linear Regression** is trained on the scaled features.
   - **Ridge Regression** is also implemented to handle multicollinearity and prevent overfitting . Ridge Regression is trained using **Polynomial Features**

4. **Model Evaluation**  
   - **Mean Squared Error (MSE)** is calculated to measure model performance.
   - **R² Score** is used to evaluate the model’s accuracy.


## 📈 Performance Comparison: Linear vs. Ridge Regression

| Model Type          | MSE (Lower is Better) | R² Score (Higher is Better) |
|---------------------|----------------------|-----------------------------|
| **Linear Regression** | 3410.87| 0.45|
| **Ridge Regression** | 2693.67| 0.50|

### 🔍 Observations:
- Ridge Regression **reduces MSE**, leading to **better prediction accuracy**.
- The **R² Score improved**, indicating a better fit to the data.
- Overfitting is addressed as Ridge Regression **prevents excessive weight magnitudes** on features.
- On the training set, **Linear Regression showed a higher R² than on the test set**, suggesting overfitting, which Ridge Regression corrected.

## 🛠️ Technologies Used

- **Python**
- **Scikit-Learn** (`sklearn`)
- **NumPy**
- **Pandas**

## 📌 How to Run This Project

1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn    matplotlib.pyplot 
   ```
   or see requirements.txt file in repo.
   
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open the `Regression Model-Diabetes Prediction(SCALED FEATURES).ipynb` file and execute the cells.


5. **View Predictions & Plots**  
   The model will generate predictions, evaluation metrics, and visualizations.

## 📌 Notes

- Feature scaling is crucial when using models like Ridge Regression.
- Ridge Regression helps in handling multicollinearity and improving model stability.
- The dataset is small, so performance can be impacted by data splitting.

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

