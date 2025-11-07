import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Предполагается, что у вас реализована модель Im2LatexModel
from Im2LatexModel import Im2LatexModel

# Список токенов (105 элементов), используемый при обучении
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

# Формируем словари для преобразования токен <-> индекс.
token_to_idx = {}
for i, token in enumerate(vocab_list):
    if token not in token_to_idx:
        token_to_idx[token] = i
idx_to_token = {i: token for i, token in enumerate(vocab_list)}


def beam_search_decode(model, image_tensor, start_token, beam_width=15, max_length=50, temperature=3.0):
    """
    Beam search декодирование с температурным масштабированием.

    Параметры:
      model         - обученная модель Im2LatexModel.
      image_tensor  - входное изображение [1, C, H, W].
      start_token   - индекс стартового токена.
      beam_width    - число кандидатов, сохраняемых на каждом шаге.
      max_length    - максимальная длина последовательности.
      temperature   - значение температурного масштабирования (больше значения сглаживают распределение сильнее).

    Возвращает последовательность токенов лучшего кандидата (список индексов).
    """
    model.eval()
    with torch.no_grad():
        features = model.encoder(image_tensor)
        # Обработка выхода encoder в зависимости от его размерности:
        if features.dim() == 3:  # [B, L, D] – усредняем по L
            mean_features = features.mean(dim=1)
        elif features.dim() == 4:  # [B, C, H, W] – применяем adaptive pooling
            mean_features = F.adaptive_avg_pool2d(features, output_size=(4, 4)).view(features.size(0), -1)
        elif features.dim() == 2:
            mean_features = features
        else:
            raise ValueError("Неожиданный формат выхода encoder: " + str(features.shape))

        h0 = torch.tanh(model.decoder.init_h(mean_features)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        hidden0 = (h0, c0)

        # Первый (начальный) кандидат
        beam = [([start_token], 0.0, hidden0)]

        for _ in range(max_length):
            new_beam = []
            for tokens, score, hidden_state in beam:
                if tokens[-1] == 0:  # если стоп-токен, оставляем кандидата без расширения
                    new_beam.append((tokens, score, hidden_state))
                    continue
                input_token = torch.tensor([[tokens[-1]]], device=image_tensor.device)
                embed = model.decoder.embedding(input_token)
                embed = model.decoder.dropout(embed)
                output, new_hidden = model.decoder.lstm(embed, hidden_state)
                output = model.decoder.dropout(output)
                logits = model.decoder.fc(output).squeeze()  # [vocab_size]
                # Применяем температурное масштабирование
                logits = logits / temperature
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    next_token = topk_indices[i].item()
                    next_log_prob = topk_log_probs[i].item()
                    new_tokens = tokens + [next_token]
                    new_score = score + next_log_prob
                    new_beam.append((new_tokens, new_score, new_hidden))
            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
            beam = new_beam
            if all(candidate[0][-1] == 0 for candidate in beam):
                break
        best_candidate = max(beam, key=lambda x: x[1])[0]
        return best_candidate


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Список путей к нескольким изображениям из обучающей выборки для эксперимента.
    image_paths = [
        r"E:\3курс\курс\img_processed\train\OffHME\00002.png",
        r"E:\3курс\курс\img_processed\train\OffHME\00003.png",
        r"E:\3курс\курс\img_processed\train\OffHME\00004.png"
        # Добавьте другие пути по необходимости.
    ]

    # Значения параметра temperature, которые будем пробовать
    temperatures = [3.0, 4.0, 5.0]

    # Преобразования, как при обучении (конвертация в оттенки серого, Resize, ToTensor & Normalize)
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Загружаем модель и состояние чекпоинта (убедитесь, что vocab_size=105)
    model = Im2LatexModel(vocab_size=105, hidden_size=256, dropout_p=0.5).to(device)
    checkpoint_path = "im2latex_checkpoint_epoch_30.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Стартовый токен (например, символ '!' используется как начало)
    start_token = token_to_idx['!']

    # Для каждого изображения и для каждого значения температуры запускаем декодирование
    for img_path in image_paths:
        print("=" * 60)
        print("Абсолютный путь:", os.path.abspath(img_path))
        if not os.path.isfile(img_path):
            print("Файл не найден, пропускаем", img_path)
            continue
        # Загружаем изображение в режиме L (оттенки серого)
        image = Image.open(img_path).convert('L')
        image_tensor = test_transforms(image).unsqueeze(0).to(device)  # [1, 1, 256, 256]

        for temp in temperatures:
            predicted_indices = beam_search_decode(model, image_tensor, start_token,
                                                   beam_width=15, max_length=50, temperature=temp)
            predicted_tokens = [idx_to_token[idx] for idx in predicted_indices]
            predicted_caption = ' '.join(predicted_tokens)
            print(f"Temperature = {temp:.1f}, Predicted Caption: {predicted_caption}")
