import pandas as pd
import random
from faker import Faker

fake = Faker()

def generate_dataset(n=1000):
    data = []
    for _ in range(n):
        idade = random.randint(18, 70)
        salario = random.randint(1200, 15000)
        emprestimo = random.randint(500, 20000)
        score = random.randint(300, 900)
        inadimplente = 1 if (salario < 3000 and emprestimo > 10000) else 0

        data.append({
            "nome": fake.name(),
            "idade": idade,
            "salario": salario,
            "valor_emprestimo": emprestimo,
            "score_credito": score,
            "inadimplente": inadimplente
        })

    return pd.DataFrame(data)
