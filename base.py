# 1. Сначала установим необходимые библиотеки
#pip install transformers torch sentencepiece accelerate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def setup_local_assistant():
    # Загружаем модель и токенизатор
    # Используем BLOOMZ-560M - относительно легкая мультиязычная модель
    model_name = "bigscience/bloomz-560m"

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Загрузка модели
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Используем float16 для экономии памяти
        low_cpu_mem_usage=True,
    )

    # Если есть GPU, используем его
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Создаем пайплайн для генерации текста
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    return generator


def get_assistant_response(generator, prompt, max_length=600):
    """
    Генерирует ответ на основе входного промпта
    """
    # Формируем промпт в формате инструкции
    formatted_prompt = f"""Задача: {prompt}
    Ответ:"""

    # Генерируем ответ
    response = generator(
        formatted_prompt,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Извлекаем сгенерированный текст
    generated_text = response[0]['generated_text']
    # Убираем исходный промпт из ответа
    answer = generated_text[len(formatted_prompt):]

    return answer.strip()


# Пример использования
if __name__ == "__main__":
    # Инициализируем ассистента
    print("Загрузка модели...")
    generator = setup_local_assistant()

    # Пример работы с документами
    test_prompt = """
show me how can i implement "builder" programming pattern in Python?

    """

    print("\nГенерация ответа...")
    response = get_assistant_response(generator, test_prompt)
    print(f"\nОтвет: {response}")