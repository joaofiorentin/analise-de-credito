import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    # Seleciona features e target
    X = df[["idade", "salario", "valor_emprestimo", "score_credito"]]
    y = df["inadimplente"]

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balanceamento com SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)

    # Split
    return train_test_split(X_res, y_res, test_size=0.2, random_state=42)
