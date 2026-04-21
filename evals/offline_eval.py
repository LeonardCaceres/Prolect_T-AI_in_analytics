# evals/offline_eval.py
import json
import pandas as pd
from langfuse import Langfuse
from openai import OpenAI

# Инициализация клиентов
langfuse = Langfuse()
client = OpenAI()

# Загрузка эталонного датасета (предполагается, что он создан в Langfuse)
dataset = langfuse.get_dataset("ab_test_benchmark")

# Варианты системных промптов для сравнения
prompt_variants = {
    "v1_standard": "Ты — AI-ассистент для A/B-тестов...",
    "v2_detailed": "Ты — опытный аналитик данных...",
    "v3_concise": "Ты — эксперт по A/B-тестированию. Давай краткие и точные ответы."
}

results = []

for item in dataset.items:
    input_data = item.input
    expected_output = item.expected_output
    
    for prompt_name, system_prompt in prompt_variants.items():
        # Вызов LLM для каждого варианта
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_data["question"]}
            ]
        )
        
        generated_output = response.choices[0].message.content
        
        # Оценка ответа с помощью LLM as a Judge
        eval_prompt = f"""
        Оцени ответ ассистента по шкале от 0 до 10.
        Вопрос: {input_data["question"]}
        Эталонный ответ: {expected_output}
        Ответ ассистента: {generated_output}
        
        Критерии оценки:
        - Точность расчетов (0-4 балла)
        - Правильность интерпретации (0-3 балла)
        - Ясность и полнота объяснения (0-3 балла)
        
        Верни только число (0-10).
        """
        
        eval_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": eval_prompt}]
        )
        
        score = float(eval_response.choices[0].message.content.strip())
        
        results.append({
            "item_id": item.id,
            "prompt_variant": prompt_name,
            "score": score
        })
        
        # Логируем результат в Langfuse
        trace = langfuse.trace(name=f"eval_{item.id}_{prompt_name}")
        trace.score(name="llm_judge_score", value=score)

# Анализ результатов
df = pd.DataFrame(results)
summary = df.groupby("prompt_variant")["score"].agg(["mean", "std", "count"])
print("Результаты сравнения вариантов промптов:")
print(summary)

# Обоснование выбора лучшего варианта
best_variant = summary["mean"].idxmax()
print(f"\nВыбран вариант '{best_variant}' со средним баллом {summary.loc[best_variant, 'mean']:.2f}")
print("Обоснование: Данный вариант показывает наилучшее сочетание точности расчетов и понятности объяснений.")
