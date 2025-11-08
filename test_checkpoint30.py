import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence

# Импортируйте классы датасета и модели согласно структуре вашего проекта
from Dataset import MathExpressionDataset
from Im2LatexModel import Im2LatexModel

# Импортируем BLEU score из nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Определяем список токенов (vocabulary). Дубликаты удаляются.
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
vocab_list = list(dict.fromkeys(vocab_list))  # Убираем дубликаты

# Создаем словарь token -> индекс
token_to_idx = {token: i for i, token in enumerate(vocab_list)}

# Создаем обратное отображение: индекс -> token (для BLEU score)
idx_to_token = {i: token for token, i in token_to_idx.items()}


def tokenize_caption(caption_dict):
    """
    Если подпись представлена в виде словаря без предобработанных токенов,
    извлекаем последовательность индексов, используя секцию "objects".

    Для токенов, где фигурные скобки экранированы (например, "\\{" или "\\}"),
    заменяем их на обычные "{" или "}".
    """
    if "objects" not in caption_dict:
        raise ValueError("Caption dictionary must contain an 'objects' key.")
    tokens = []
    for obj in caption_dict["objects"]:
        symbol = obj.get("symbol", "").strip()
        if symbol in {"\\{", "\\}"}:
            symbol = symbol[1:]  # заменяем "\\{" на "{" и "\\}" на "}"
        if symbol == "":
            continue
        if symbol not in token_to_idx:
            raise ValueError(f"Token '{symbol}' not found in vocabulary!")
        tokens.append(token_to_idx[symbol])
    return torch.tensor(tokens, dtype=torch.long)


def my_collate_fn(batch):
    """
    Объединяет элементы батча. Каждый элемент — пара (image, caption).

    Если подпись представлена в виде словаря, сперва проверяются ключи 'tokens' или 'caption'.
    Если их нет, применяется функция tokenize_caption для извлечения последовательности индексов по ключу "objects".

    Все изображения приводятся к CPU (для корректной работы pin_memory),
    а подписи дополняются до одинаковой длины с помощью pad_sequence.
    """
    images = [item[0].cpu() if isinstance(item[0], torch.Tensor) else item[0] for item in batch]
    captions = [item[1] for item in batch]
    images = torch.stack(images, dim=0)

    processed_captions = []
    for caption in captions:
        if isinstance(caption, dict):
            if "tokens" in caption:
                tokens = caption["tokens"]
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens, dtype=torch.long)
                processed_captions.append(tokens)
            elif "caption" in caption:
                tokens = caption["caption"]
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens, dtype=torch.long)
                processed_captions.append(tokens)
            else:
                tokens = tokenize_caption(caption)
                processed_captions.append(tokens)
        elif isinstance(caption, torch.Tensor):
            processed_captions.append(caption.cpu())
        elif isinstance(caption, str):
            # Если подпись представлена строкой, можно реализовать другую логику токенизации.
            # Здесь, например, превращаем каждый символ в его ASCII-код.
            processed_captions.append(torch.tensor([ord(c) for c in caption], dtype=torch.long))
        else:
            processed_captions.append(torch.tensor(caption, dtype=torch.long))

    captions_padded = pad_sequence(processed_captions, batch_first=True, padding_value=0)
    return images, captions_padded


def test_model(model, test_loader, device):
    model.eval()  # Переводим модель в режим тестирования
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    exact_match_count = 0  # Счетчик строго правильных (exact match) предсказаний
    total_examples = len(test_loader.dataset)  # Общее число тестовых примеров

    bleu_scores = []  # Список для накопления BLEU score по каждому примеру
    smooth_fn = SmoothingFunction().method1

    with torch.no_grad():
        for images, captions in test_loader:
            images = images.to(device)
            captions = captions.to(device)

            # Формируем входную последовательность и целевую последовательность
            input_captions = captions[:, :-1]  # без последнего токена
            target_captions = captions[:, 1:]  # начиная со второго токена

            outputs = model(images, input_captions)  # [batch, seq_len, vocab_size]

            # Расчет функции потерь
            outputs_flat = outputs.reshape(-1, outputs.size(-1))  # [batch*seq_len, vocab_size]
            target_flat = target_captions.reshape(-1)  # [batch*seq_len]
            loss = criterion(outputs_flat, target_flat)
            test_loss += loss.item()

            # Получаем предсказанные токены (индексы)
            predicted_tokens = outputs.argmax(dim=-1)  # [batch, seq_len]

            # Для каждого примера вычисляем exact match и BLEU score
            for pred_seq, true_seq in zip(predicted_tokens, target_captions):
                if torch.equal(pred_seq, true_seq):
                    exact_match_count += 1

                # Преобразуем последовательности в списки токенов (удаляем паддинг 0)
                pred_list = [idx_to_token[token.item()] for token in pred_seq if token.item() != 0]
                true_list = [idx_to_token[token.item()] for token in true_seq if token.item() != 0]

                # Вычисляем BLEU score для примера (используем сглаживание)
                bleu = sentence_bleu([true_list], pred_list, smoothing_function=smooth_fn)
                bleu_scores.append(bleu)

    avg_test_loss = test_loss / len(test_loader)
    exact_match_accuracy = (exact_match_count / total_examples) * 100
    avg_bleu = (sum(bleu_scores) / len(bleu_scores)) * 100  # умножаем на 100 для процентов

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.2f}%")
    print(f"Average BLEU Score: {avg_bleu:.2f}%")


if __name__ == "__main__":
    # Определяем устройство (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Пути к тестовому набору изображений и аннотациям
    test_img_folder = r"E:\3курс\курс\img_processed\test"
    base_annotations_path = r"C:\Users\Admin\Desktop\KursProject"
    test_annotation_file = os.path.join(base_annotations_path, "test.json")

    # Преобразования для изображений: Resize, ToTensor и Normalize
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Загружаем тестовый датасет и создаем DataLoader с использованием my_collate_fn
    test_dataset = MathExpressionDataset(test_img_folder, test_annotation_file, data_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,  # num_workers=0 для Windows
        pin_memory=True,
        collate_fn=my_collate_fn
    )

    # Загружаем модель и чекпоинт 30-й эпохи
    model = Im2LatexModel(vocab_list, hidden_size=256, dropout_p=0.5).to(device)
    checkpoint_path = "im2latex_checkpoint_epoch_30.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Запускаем тестирование модели
    test_model(model, test_loader, device)
