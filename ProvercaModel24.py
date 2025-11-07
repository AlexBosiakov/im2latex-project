import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from Dataset import MathExpressionDataset
from Im2LatexModel import Im2LatexModel
from train_im2latex import my_collate_fn

# Пути к данным
val_img_folder = r"E:\3курс\курс\img_processed\val"
base_annotations_path = r"C:\Users\Admin\Desktop\KursProject"
val_annotation_file = os.path.join(base_annotations_path, "val.json")

# Преобразования для валидационного датасета (без аугментации)
val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Загрузка датасета и DataLoader'а
val_dataset = MathExpressionDataset(val_img_folder, val_annotation_file, val_transforms)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем модель из 24-го чекпоинта
checkpoint_path = "im2latex_checkpoint_epoch_24.pth"
model = Im2LatexModel(vocab_size=105, hidden_size=256).to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Переводим модель в режим проверки

# Список токенов (вместо token_to_idx)
vocab_list = [
    '!', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', '=', 'A', 'B', 'C', 'COMMA', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P',
    'R', 'S', 'T', 'V', 'X', 'Y', '[', '\\Delta', '\\alpha', '\\beta', '\\cos', '\\div',
    '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\lambda',
    '\\ldots', '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pi', '\\pm',
    '\\prime', '\\rightarrow', '\\sigma', '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta',
    '\\times', '{', '}', ']', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}'
]
token_to_idx = {token: i for i, token in enumerate(vocab_list)}
idx_to_token = {i: token for token, i in token_to_idx.items()}  # ✅ Исправлено

# Функция декодирования предсказанных токенов
def decode_tokens(tokens):
    return " ".join([idx_to_token.get(idx, "[UNK]") for idx in tokens])  # [UNK] если токен не найден

# Функция для отображения предсказаний
def show_prediction(image_tensor, predicted, real):
    image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title(f"Predicted: {predicted}\nReal: {real}", fontsize=10)
    plt.show()

# Проверка на 10 примерах
for batch_idx, (images, captions) in enumerate(val_loader):
    images = images.to(device)

    with torch.no_grad():
        predicted_outputs = model(images, captions[:, :-1])  # Teacher Forcing

    predicted_tokens = torch.argmax(predicted_outputs, dim=-1).cpu().numpy()
    predicted_formula = decode_tokens(predicted_tokens[0])
    real_formula = decode_tokens(captions[0].cpu().numpy())

    # Вывод текстового результата
    print(f"Real LaTeX:    {real_formula}")
    print(f"Predicted LaTeX: {predicted_formula}")
    print("-" * 50)

    # Визуализация изображения и формул
    show_prediction(images, predicted_formula, real_formula)

    # Проверим 10 изображений и остановимся
    if batch_idx >= 9:
        break
