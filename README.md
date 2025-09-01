# Projeto: Predição de Risco de Crédito

Este projeto implementa um pipeline de Ciência de Dados para predição de inadimplência, utilizando dados sintéticos gerados com a biblioteca Faker.

## Estrutura de Pastas
- data/ : armazenar datasets
- notebooks/ : contém o notebook principal (credito_risco.ipynb)
- src/ : scripts auxiliares (pré-processamento, funções, etc.)
- results/ : resultados, gráficos e métricas

## Como usar
1. Instale as dependências:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn faker
   ```

2. Abra o notebook:
   ```bash
   jupyter notebook notebooks/credito_risco.ipynb
   ```

3. Execute célula por célula para treinar e avaliar os modelos.
