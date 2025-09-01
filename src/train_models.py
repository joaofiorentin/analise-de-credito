from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    models = {}

    # Regressão Logística
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    models["Logistic Regression"] = log_reg

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    return models
