import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from data_preprocessing import load_and_process_data

def train_models(file_path):
    X_men, y_men, X_women, y_women = load_and_process_data(file_path)
    
    X_train_men, X_test_men, y_train_men, y_test_men = train_test_split(X_men, y_men, test_size=0.2, random_state=42)

    X_train_women, X_test_women, y_train_women, y_test_women = train_test_split(X_women, y_women, test_size=0.2, random_state=42)
    
    model_men = GaussianNB()
    model_men.fit(X_train_men, y_train_men)
    y_pred_men = model_men.predict(X_test_men)
    
    # Evaluate Naive Bayes model for men
    accuracy_men = accuracy_score(y_test_men, y_pred_men)
    sensitivity_men = recall_score(y_test_men, y_pred_men)
    specificity_men = precision_score(y_test_men, y_pred_men, pos_label=0)
    print(f'Men Model Accuracy: {accuracy_men}')
    print(f'Men Model Sensitivity (Recall): {sensitivity_men}')
    print(f'Men Model Specificity (Precision for non-T2DM): {specificity_men}')
    
    # Train Logistic Regression model for women
    model_women = LogisticRegression(max_iter=1000)
    model_women.fit(X_train_women, y_train_women)
    y_pred_women = model_women.predict(X_test_women)
    
    # Evaluate Logistic Regression model for women
    accuracy_women = accuracy_score(y_test_women, y_pred_women)
    sensitivity_women = recall_score(y_test_women, y_pred_women)
    specificity_women = precision_score(y_test_women, y_pred_women, pos_label=0)
    print(f'Women Model Accuracy: {accuracy_women}')
    print(f'Women Model Sensitivity (Recall): {sensitivity_women}')
    print(f'Women Model Specificity (Precision for non-T2DM): {specificity_women}')
    
    # Save the models
    joblib.dump(model_men, 'MLmodels/naive_bayes_men.pkl')
    joblib.dump(model_women, 'MLmodels/logistic_regression_women.pkl')
