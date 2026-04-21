# ui/app.py
import streamlit as st
import requests
import json

# Конфигурация n8n Webhook
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/abtest-assistant"  # Замените на ваш URL вебхука

st.set_page_config(page_title="A/B Test Assistant", layout="wide")
st.title("🧪 Ассистент для проведения A/B тестов")

# Инициализация истории сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ввод пользователя
if prompt := st.chat_input("Опишите вашу идею для A/B теста..."):
    # Добавляем сообщение пользователя в историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Отправляем запрос в n8n
    with st.chat_message("assistant"):
        with st.spinner("Ассистент думает..."):
            try:
                response = requests.post(
                    N8N_WEBHOOK_URL,
                    json={"message": prompt, "session_id": "streamlit_ui"},
                    timeout=60
                )
                response.raise_for_status()
                assistant_response = response.json().get("output", "Извините, произошла ошибка.")
            except Exception as e:
                assistant_response = f"Ошибка связи с ассистентом: {e}"
            
            st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Боковая панель с примерами и параметрами
with st.sidebar:
    st.header("Параметры теста по умолчанию")
    power = st.slider("Мощность теста (Power)", 0.7, 0.95, 0.8, 0.01)
    alpha = st.slider("Уровень значимости (Alpha)", 0.01, 0.1, 0.05, 0.01)
    
    st.divider()
    st.header("Примеры запросов")
    st.markdown("""
    - "Хочу увеличить конверсию в покупку. Какой размер выборки нужен?"
    - "Мы изменили цвет кнопки. Контроль: 1000 показов, 50 кликов. Тест: 1000 показов, 65 кликов. Есть ли эффект?"
    - "Сформулируй гипотезу для тестирования новой рекомендательной системы."
    """)
