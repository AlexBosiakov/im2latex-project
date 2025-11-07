import os
import json
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# Импорт модели и датасета. Убедитесь, что файлы Im2LatexModel.py и Dataset.py находятся в рабочей директории.
from Im2LatexModel import Im2LatexModel
from Dataset import MathExpressionDataset

# Исходный список токенов (105 уникальных символов), как использовался при обучении.
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

# Важно: не удаляем дубликаты, чтобы размер словаря оставался равным 105.
token_to_idx = {token: i for i, token in enumerate(vocab_list)}
idx_to_token = {i: token for token, i in token_to_idx.items()}


def beam_search_decode(model, image, device, beam_width=3, max_length=50, start_token=0, end_token=None):
    """
    Реализует beam search decoding для одного изображения.

    Аргументы:
      model: обученная модель Im2LatexModel.
      image: тензор изображения размером [in_channels, H, W].
      device: устройство (CPU или GPU).
      beam_width: количество кандидатов для расширения на каждом шаге.
      max_length: максимальная длина генерируемой последовательности.
      start_token: индекс стартового токена (например, 0).
      end_token: индекс конечного токена (если используется).

    Возвращает:
      Лучшую последовательность токенов (без стартового токена).
    """
    model.eval()
    with torch.no_grad():
        # Подготавливаем изображение: добавляем измерение батча
        image = image.unsqueeze(0).to(device)
        features = model.encoder(image)
        mean_features = features.mean(dim=1)  # средние признаки для инициализации LSTM
        h0 = torch.tanh(model.decoder.init_h(mean_features)).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        # Каждый элемент beam: (sequence, h, c, cumulative_log_prob)
        beam = [([start_token], h0, c0, 0.0)]

        for _ in range(max_length):
            new_beam = []
            for seq, h, c, cum_log_prob in beam:
                if end_token is not None and seq[-1] == end_token:
                    new_beam.append((seq, h, c, cum_log_prob))
                    continue

                token_input = torch.tensor([[seq[-1]]], dtype=torch.long).to(device)
                embedding = model.decoder.embedding(token_input)  # [1,1,hidden_size]
                output, (new_h, new_c) = model.decoder.lstm(embedding, (h, c))
                logits = model.decoder.fc(output.squeeze(1))  # [1, vocab_size]
                log_probs = F.log_softmax(logits, dim=1)

                top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
                top_log_probs = top_log_probs.squeeze(0)
                top_indices = top_indices.squeeze(0)

                for log_prob, token_idx in zip(top_log_probs, top_indices):
                    new_seq = seq + [token_idx.item()]
                    new_cum_log_prob = cum_log_prob + log_prob.item()
                    new_beam.append((new_seq, new_h, new_c, new_cum_log_prob))

            # Сохраняем beam_width лучших кандидатов
            beam = sorted(new_beam, key=lambda x: x[3], reverse=True)[:beam_width]
            if end_token is not None and all(seq[-1] == end_token for seq, _, _, _ in beam):
                break

        best_seq = sorted(beam, key=lambda x: x[3], reverse=True)[0][0]
        return best_seq[1:]  # Отбрасываем стартовый токен


def levenshtein_distance(s1, s2):
    """
    Вычисляет Левенштейново расстояние между двумя строками.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalized_similarity(s1, s2):
    """
    Вычисляет нормализованное Левенштейново расстояние и возвращает меру сходства в диапазоне [0, 1].
    При полном совпадении similarity = 1.
    """
    if not s1 and not s2:
        return 1.0
    lev_distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    similarity = 1 - (lev_distance / max_len)
    return similarity


def process_ground_truth(caption):
    """
    Преобразует подпись из ground truth в строку для сравнения.
    Если подпись представлена в виде словаря, пытается извлечь значение ключа "caption" или объединить символы из "objects".
    """
    if isinstance(caption, dict):
        if "caption" in caption:
            return str(caption["caption"]).strip()
        elif "objects" in caption:
            tokens = []
            for obj in caption["objects"]:
                symbol = obj.get("symbol", "").strip()
                if symbol in {"\\{", "\\}"}:
                    symbol = symbol[1:]
                if symbol:
                    tokens.append(symbol)
            return " ".join(tokens)
        else:
            return str(caption).strip()
    else:
        return str(caption).strip()


def test_model_with_beam_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Преобразования для тестовых изображений
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Пути к тестовому набору
    test_img_folder = r"E:\3курс\курс\img_processed\test"
    base_annotations_path = r"C:\Users\Admin\Desktop\KursProject"
    test_annotation_file = os.path.join(base_annotations_path, "test.json")

    test_dataset = MathExpressionDataset(test_img_folder, test_annotation_file, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Создаем модель с тем же размером словаря (105 токенов)
    VOCAB_SIZE = len(vocab_list)
    model = Im2LatexModel(vocab_size=VOCAB_SIZE, hidden_size=256).to(device)

    checkpoint_path = "im2latex_checkpoint_epoch_30.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Загружен чекпоинт: {checkpoint_path}")

    total_similarity = 0.0
    sample_count = 0

    print("Начало инференса с beam search декодированием и вычислением Soft metric:")
    for idx, (images, captions) in enumerate(test_loader):
        images = images.to(device)
        image = images[0]  # [in_channels, H, W]

        # Генерация подписи с использованием beam search
        generated_token_seq = beam_search_decode(model, image, device,
                                                 beam_width=3, max_length=50,
                                                 start_token=0, end_token=None)
        generated_caption = " ".join([idx_to_token[token] for token in generated_token_seq])
        gt_caption = process_ground_truth(captions[0])

        sim = normalized_similarity(generated_caption, gt_caption)
        total_similarity += sim
        sample_count += 1

        print(f"Изображение {idx + 1}:")
        print(f"  Сгенерированная подпись: {generated_caption}")
        print(f"  Ground truth: {gt_caption}")
        print(f"  Схожесть (soft metric): {sim:.4f}")
        print("-" * 60)

        # Для примера обработаем первые 5 изображений
        if idx == 4:
            break

    avg_similarity = total_similarity / sample_count if sample_count > 0 else 0
    print(f"\nСредняя схожесть (Soft metric) по тестовому набору: {avg_similarity:.4f}")


if __name__ == "__main__":
    test_model_with_beam_search()
