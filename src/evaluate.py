from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_models(models, X_test, y_test, results_path="results"):
    # Cria a pasta results se não existir
    os.makedirs(results_path, exist_ok=True)

    for name, model in models.items():
        print(f"\n📊 Avaliação do modelo: {name}")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Classification Report
        report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(report)
        with open(os.path.join(results_path, f"classification_report_{name}.txt"), "w") as f:
            f.write(report)

        # Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusão - {name}")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.savefig(os.path.join(results_path, f"confusion_matrix_{name}.png"))
        plt.show()

        # Curva ROC
        plt.figure()
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)
        plt.savefig(os.path.join(results_path, f"roc_curve_{name}.png"))
        plt.show()

        # AUC ROC
        auc = roc_auc_score(y_test, y_proba)
        print(f"AUC ROC: {auc:.4f}")
        with open(os.path.join(results_path, f"auc_roc_{name}.txt"), "w") as f:
            f.write(f"AUC ROC: {auc:.4f}")
