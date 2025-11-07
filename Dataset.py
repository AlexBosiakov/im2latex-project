import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MathExpressionDataset(Dataset):
    def __init__(self, img_folder, annotation_file, transform=None):
        """
        img_folder - корневая папка с изображениями для набора (например, train)
        annotation_file - путь к JSON-файлу с аннотациями
        transform - преобразования для изображения (например, ToTensor, Resize и т.д.)
        """
        self.img_folder = img_folder
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        # Фильтруем ключи, исключая временные файлы, начинающиеся с "~$"
        self.file_names = [key for key in self.annotations.keys() if not key.startswith("~$")]
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Получаем имя (ключ) аннотации, например "001-equation000.lg"
        key = self.file_names[idx]
        # Формируем имя изображения: заменяем расширение .lg на .png
        img_file = os.path.splitext(key)[0] + ".png"

        # Рекурсивно ищем изображение в self.img_folder (включая вложенные папки)
        found_path = None
        for root, _, files in os.walk(self.img_folder):
            if img_file in files:
                found_path = os.path.join(root, img_file)
                break

        if found_path is None:
            raise FileNotFoundError(f"Файл не найден: {img_file} в {self.img_folder}")

        # Загружаем изображение. Используйте 'L' для grayscale или 'RGB' по необходимости.
        image = Image.open(found_path).convert('L')
        if self.transform:
            image = self.transform(image)

        # Из аннотаций получаем подпись (caption) для данного примера
        caption = self.annotations[key]
        return image, caption


if __name__ == '__main__':
    # Определяем преобразования для изображений
    data_transforms = transforms.Compose([
        # Добавьте другие преобразования по необходимости (Resize, Normalize и т.д.)
        transforms.ToTensor()
    ])

    # Пути к наборам изображений (train, val, test)
    train_img_folder = r"E:\3курс\курс\img_processed\train"
    val_img_folder = r"E:\3курс\курс\img_processed\val"
    test_img_folder = r"E:\3курс\курс\img_processed\test"

    # Пути к JSON-антнотациям
    base_annotations_path = r"C:\Users\Admin\Desktop\KursProject"
    train_annotation_file = os.path.join(base_annotations_path, "train.json")
    val_annotation_file = os.path.join(base_annotations_path, "val.json")
    test_annotation_file = os.path.join(base_annotations_path, "test.json")

    # Создаем датасеты для каждого набора
    train_dataset = MathExpressionDataset(train_img_folder, train_annotation_file, data_transforms)
    val_dataset = MathExpressionDataset(val_img_folder, val_annotation_file, data_transforms)
    test_dataset = MathExpressionDataset(test_img_folder, test_annotation_file, data_transforms)

    # Создаем DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Пример использования DataLoader'а: перевод изображений на устройство (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_idx, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        print(f"Batch {batch_idx}: images shape = {images.shape}")
        print(f"Sample caption: {captions[0]}")
        break
