# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install transformers accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


def check_gpu_setup():
    """Проверка настройки GPU и доступной памяти"""
    print("=== Проверка GPU ===")

    if torch.cuda.is_available():
        print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
        print(f"Версия CUDA: {torch.version.cuda}")

        # Проверка доступной памяти
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        reserved_mem = torch.cuda.memory_reserved(0) / 1024 ** 3
        allocated_mem = torch.cuda.memory_allocated(0) / 1024 ** 3
        free_mem = total_mem - allocated_mem

        print(f"\nПамять GPU:")
        print(f"Всего: {total_mem:.2f} GB")
        print(f"Свободно: {free_mem:.2f} GB")
        print(f"Зарезервировано: {reserved_mem:.2f} GB")
        print(f"Использовано: {allocated_mem:.2f} GB")

        return True
    else:
        print("GPU не обнаружен!")
        return False


def load_model_with_gpu_optimizations(model_name):
    """Загрузка модели с оптимизациями для GPU"""
    print(f"\n=== Загрузка модели {model_name} ===")

    # Очистка памяти GPU перед загрузкой
    gc.collect()
    torch.cuda.empty_cache()

    # Оптимизации для экономии памяти
    config = {
        "torch_dtype": torch.float16,  # Использование половинной точности
        "low_cpu_mem_usage": True,  # Оптимизация использования CPU памяти
        "device_map": "auto",  # Автоматическое распределение на доступные устройства
    }

    try:
        # Загрузка модели с оптимизациями
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **config
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Модель успешно загружена на GPU")

        # Проверка размещения модели
        print(f"\nРазмещение модели: {model.device}")

        return model, tokenizer

    except Exception as e:
        print(f"Ошибка при загрузке модели: {str(e)}")
        return None, None


def generate_test_response(model, tokenizer, prompt="Привет! Как дела?", max_length=100):
    """Тестовая генерация текста для проверки работы GPU"""
    print("\n=== Тестовая генерация ===")

    try:
        # Замер времени генерации
        import time
        start_time = time.time()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = time.time() - start_time

        print(f"Время генерации: {generation_time:.2f} секунд")
        print(f"Ответ: {response}")

        # Проверка использования памяти после генерации
        print(f"\nИспользование GPU после генерации: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")

        return response

    except Exception as e:
        print(f"Ошибка при генерации: {str(e)}")
        return None


if __name__ == "__main__":
    # Проверка настройки GPU
    if check_gpu_setup():
        # Загружаем небольшую модель для теста
        model_name = "microsoft/phi-2"  # ~2.7GB VRAM
        model, tokenizer = load_model_with_gpu_optimizations(model_name)

        if model is not None:
            # Тестовая генерация
            generate_test_response(model, tokenizer)