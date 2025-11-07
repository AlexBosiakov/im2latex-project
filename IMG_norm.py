from PIL import Image
import os
import numpy as np

# Задаем пути к исходной и выходной папкам.
base_folder = r"E:\3курс\курс\img"
output_folder = r"E:\3курс\курс\img_processed"

# Создаем основную выходную папку, если её ещё нет.
os.makedirs(output_folder, exist_ok=True)

# Обрабатываем наборы данных: train, val, test.
for dataset_type in ["train", "val", "test"]:
    input_path = os.path.join(base_folder, dataset_type)

    # Рекурсивно обходим все вложенные папки.
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            # Обрабатываем только файлы с расширением .png (без учета регистра).
            if filename.lower().endswith(".png"):
                img_path = os.path.join(root, filename)

                # Определяем относительный путь от base_folder, чтобы сохранить вложенную структуру.
                relative_path = os.path.relpath(img_path, base_folder)
                save_path = os.path.join(output_folder, relative_path)

                # Создаем директории для сохранения, если они ещё не существуют.
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                try:
                    # Открываем изображение через Pillow и переводим его в градации серого.
                    img = Image.open(img_path).convert('L')

                    # Используем фильтр LANCZOS (в более новых версиях Pillow используется Image.Resampling.LANCZOS).
                    try:
                        resample_filter = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample_filter = Image.LANCZOS

                    # Масштабируем изображение до 256x256 пикселей.
                    img_resized = img.resize((256, 256), resample_filter)

                    # Преобразуем изображение в массив NumPy, нормализуем значения пикселей (0–1).
                    np_img = np.array(img_resized).astype(np.float32) / 255.0

                    # Перед сохранением возвращаем значения к диапазону 0–255.
                    output_img = (np_img * 255).astype(np.uint8)
                    Image.fromarray(output_img).save(save_path)

                except Exception as e:
                    print(f"Ошибка при обработке {img_path}: {e}")
