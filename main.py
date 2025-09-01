from src.generate_data import generate_dataset
from src.preprocessing import preprocess_data
from src.train_models import train_models
from src.evaluate import evaluate_models
import os

def main():
    results_path = "results"
    os.makedirs(results_path, exist_ok=True)

    # 1. Gerar dados
    print("🔹 Gerando dados sintéticos...")
    df = generate_dataset(1000)

    # 2. Pré-processamento
    print("🔹 Pré-processando os dados...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 3. Treinamento dos modelos
    print("🔹 Treinando modelos...")
    models = train_models(X_train, y_train)

    # 4. Avaliação e salvar resultados
    print("🔹 Avaliando modelos...")
    evaluate_models(models, X_test, y_test, results_path=results_path)

if __name__ == "__main__":
    main()
