# fraud_card_prediction
```markdown
# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning algorithms. The dataset contains transactions made by European cardholders in September 2013, and it consists of a wide range of features extracted via PCA (Principal Component Analysis), along with a label indicating whether a transaction is fraudulent or not.

## Table of Contents

1. [Dataset](#dataset)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Features](#features)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Contributing](#contributing)
8. [License](#license)

## Dataset

- **Source**: The dataset is obtained from Kaggle. You can download it from the following link:
  [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  
- **Description**: The dataset consists of 284,807 transactions, of which 492 are fraudulent. It contains 31 features:
  - `Time`: Seconds elapsed between this transaction and the first transaction.
  - `V1` to `V28`: Principal components obtained via PCA.
  - `Amount`: The transaction amount.
  - `Class`: The label (1 indicates fraud, 0 indicates a legitimate transaction).

**Note**: The dataset is highly imbalanced, with only 0.17% of transactions being fraudulent.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ashwadhama2004/credit-card-fraud-detection.git
   ```
2. Navigate to the project directory:
   ```bash
   cd credit-card-fraud-detection
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Load the dataset:
   ```python
   import pandas as pd
   dataset = pd.read_csv("creditcard.csv")
   ```

2. Train the model using the code provided in the script:
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier

   X = dataset.drop('Class', axis=1)
   y = dataset['Class']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   ```

3. Evaluate the model:
   ```python
   from sklearn.metrics import accuracy_score
   y_pred = model.predict(X_test)
   print("Accuracy: ", accuracy_score(y_test, y_pred))
   ```

## Features

- **Data Preprocessing**: Handling missing values, standardizing features, and handling class imbalance.
- **Feature Scaling**: Using `StandardScaler` to normalize the features.
- **Machine Learning Models**:
  - Logistic Regression
  - Random Forest Classifier
- **Evaluation Metrics**:
  - Accuracy
  - Precision, Recall, F1-Score

## Model Training

- The model is trained using the `RandomForestClassifier` from the `scikit-learn` library.
- A train-test split of 80-20 is used.
- The model is trained to classify each transaction as either fraudulent or legitimate.

## Evaluation

- The model is evaluated based on accuracy, precision, recall, and F1-score.
- The results are obtained by testing the model on unseen data using the test set.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
