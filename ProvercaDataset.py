import os
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import json
from Dataset import MathExpressionDataset
img_processed_folder = r"E:\3курс\курс\img_processed"
base_annotations_path = r"C:\Users\Admin\Desktop\KursProject"
train_annotation_file = os.path.join(base_annotations_path, "train.json")
data_transforms = transforms.Compose([
    transforms.ToTensor()
])

dataset = MathExpressionDataset(
    img_folder=img_processed_folder,
    annotation_file=train_annotation_file,
    transform=data_transforms
)

print("Количество образцов в датасете:", len(dataset))


# Загружаем и проверяем первый образец
img, target = dataset[0]
print("Тип изображения:", type(img))
print("Размер изображения:", img.shape)
print("Аннотация:", target)

# Визуализация изображения: переносим тензор на CPU перед вызовом imshow
plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
plt.title("Пример изображения из датасета")
plt.axis('off')
plt.show()