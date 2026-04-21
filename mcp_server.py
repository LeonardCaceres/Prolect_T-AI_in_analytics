# mcp_server.py
from fastmcp import FastMCP
import math
import json
import os
from scipy import stats
from typing import Any
from langfuse import Langfuse

# Инициализация Langfuse (для логирования)
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="http://localhost:3000"  # локальный адрес Langfuse
)

# Инициализация MCP сервера
mcp = FastMCP("ABTest Assistant")

# --- Вспомогательные функции для расчетов ---
def calculate_sample_size_conversion(baseline_rate: float, mde: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Рассчитывает размер выборки для теста конверсий (пропорций)."""
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Двусторонний тест
    z_beta = stats.norm.ppf(power)
    
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    
    p_pooled = (p1 + p2) / 2
    se = math.sqrt(2 * p_pooled * (1 - p_pooled))
    
    n = ((z_alpha + z_beta)**2 * (p1*(1-p1) + p2*(1-p2))) / (mde * baseline_rate)**2
    return math.ceil(n)

def perform_ztest(success_a: int, trials_a: int, success_b: int, trials_b: int, alpha: float = 0.05) -> dict:
    """Выполняет z-тест для двух пропорций."""
    p_a = success_a / trials_a
    p_b = success_b / trials_b
    
    p_pooled = (success_a + success_b) / (trials_a + trials_b)
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/trials_a + 1/trials_b))
    
    z_score = (p_b - p_a) / se if se != 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Двусторонний тест
    
    return {
        "p_value": round(p_value, 4),
        "z_score": round(z_score, 4),
        "significant": p_value < alpha,
        "effect_size": round((p_b - p_a) / p_a * 100, 2) if p_a != 0 else 0
    }

# Декоратор для логирования вызовов инструментов в Langfuse
def log_to_langfuse(tool_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            trace = langfuse.trace(name=tool_name)
            try:
                result = func(*args, **kwargs)
                trace.generation(name=f"{tool_name}_output", model="gpt-4o", output=result)
                trace.score(name="tool_call_success", value=1.0)
                return result
            except Exception as e:
                trace.score(name="tool_call_success", value=0.0, comment=str(e))
                raise e
        return wrapper
    return decorator

# --- Инструменты MCP (Tools) ---
# Требование: > 3 тулов. У нас их 4.

@mcp.tool()
@log_to_langfuse("formulate_hypothesis")
def formulate_hypothesis(business_idea: str, metric_name: str = "conversion_rate") -> str:
    """
    Формулирует строгую статистическую гипотезу (H0 и H1) на основе бизнес-идеи.
    """
    # В реальности здесь будет вызов LLM, но для демонстрации генерируем шаблон.
    return f"""
    **Нулевая гипотеза (H₀):** Изменение не влияет на метрику '{metric_name}'. Разницы между контрольной и тестовой группой нет.
    **Альтернативная гипотеза (H₁):** Изменение статистически значимо влияет на метрику '{metric_name}'. 
    **Бизнес-идея:** {business_idea}
    """

@mcp.tool()
@log_to_langfuse("calculate_sample_size")
def calculate_sample_size(baseline_conversion: float, minimum_detectable_effect: float, power: float = 0.8, significance_level: float = 0.05) -> str:
    """
    Рассчитывает необходимый размер выборки для A/B-теста.
    """
    n = calculate_sample_size_conversion(baseline_conversion, minimum_detectable_effect, power, significance_level)
    return f"Для проведения A/B-теста необходимо **{n}** участников в каждой группе (всего **{2*n}**)."

@mcp.tool()
@log_to_langfuse("analyze_results")
def analyze_results(control_success: int, control_trials: int, test_success: int, test_trials: int, alpha: float = 0.05) -> str:
    """
    Анализирует результаты A/B-теста и выдает интерпретацию.
    """
    result = perform_ztest(control_success, control_trials, test_success, test_trials, alpha)
    
    if result["significant"]:
        return f"""
        **Результат A/B-теста:**
        - p-value: {result['p_value']}
        - **Статистически значимое различие обнаружено!**
        - Относительный прирост: {result['effect_size']}%
        **Интерпретация:** Можно отклонить нулевую гипотезу. Изменение с высокой вероятностью приводит к улучшению метрики.
        """
    else:
        return f"""
        **Результат A/B-теста:**
        - p-value: {result['p_value']}
        - Статистически значимого различия **не обнаружено**.
        - Относительный прирост: {result['effect_size']}%
        **Интерпретация:** Недостаточно доказательств, чтобы отклонить нулевую гипотезу. Рекомендуется оставить текущую версию или провести более мощный тест.
        """

@mcp.tool()
@log_to_langfuse("check_sample_size_sufficiency")
def check_sample_size_sufficiency(planned_size: int, actual_size: int) -> str:
    """
    Проверяет, достаточен ли фактически набранный размер выборки.
    """
    if actual_size >= planned_size:
        return f"✅ Фактический размер выборки ({actual_size}) достаточен (план: {planned_size}). Результаты теста надежны."
    else:
        return f"⚠️ Внимание: Фактический размер выборки ({actual_size}) меньше запланированного ({planned_size}). Результаты теста могут быть недостоверны из-за недостаточной мощности."

if __name__ == "__main__":
    mcp.run()
