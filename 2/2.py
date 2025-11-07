import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


#################################
# Encoder на основе ResNet18
#################################
class ResNetEncoder(nn.Module):
    def __init__(self, encoded_size=512):
        """
        Параметры:
          encoded_size - размерность выходного вектора признаков.
        """
        super(ResNetEncoder, self).__init__()
        # Загружаем ResNet18, предварительно обученную на ImageNet
        resnet = models.resnet18(pretrained=True)
        # Убираем последние два слоя (последние conv слои, adaptive pooling и fc)
        # оставляем все слои до последнего convolution блока.
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # выход: [B, 512, H', W']
        # Применяем adaptive average pooling, чтобы получить фиксированный размер карты признаков.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Линейное преобразование для получения вектора нужной размерности.
        self.fc = nn.Linear(512, encoded_size)

    def forward(self, x):
        """
        x - входное изображение (размерность: [B, C, H, W]).
        Возвращает: вектор признаков размерности [B, encoded_size].
        """
        features = self.features(x)  # [B, 512, H', W']
        pooled = self.adaptive_pool(features)  # [B, 512, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, 512]
        encoded = self.fc(pooled)  # [B, encoded_size]
        return encoded


#################################
# LSTM-декодер
#################################
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout_p=0.5):
        """
        Параметры:
          vocab_size - размер словаря (количество токенов);
          embed_size - размерность эмбеддингов для токенов;
          hidden_size - размер скрытого состояния LSTM;
          dropout_p - вероятность dropout.
        """
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, captions, hidden):
        """
        captions - входная последовательность токенов [B, seq_len]
        hidden - начальное скрытое состояние LSTM (tuple: (h0, c0)).
        Возвращает:
          logits размером [B, seq_len, vocab_size] и обновлённое hidden.
        """
        embeds = self.embedding(captions)  # [B, seq_len, embed_size]
        embeds = self.dropout(embeds)
        outputs, hidden = self.lstm(embeds, hidden)  # outputs: [B, seq_len, hidden_size]
        outputs = self.dropout(outputs)
        logits = self.fc(outputs)  # [B, seq_len, vocab_size]
        return logits, hidden


#################################
# Общая модель Im2LatexModel
#################################
class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, encoded_size=512, embed_size=256, hidden_size=256, dropout_p=0.5):
        """
        Параметры:
          vocab_size   - размер словаря токенов;
          encoded_size - размерность вектора признаков от энкодера;
          embed_size   - размер эмбеддингов для токенов;
          hidden_size  - размер скрытого состояния LSTM;
          dropout_p    - dropout (вероятность).
        """
        super(Im2LatexModel, self).__init__()
        self.encoder = ResNetEncoder(encoded_size=encoded_size)
        self.decoder = LSTMDecoder(vocab_size, embed_size, hidden_size, dropout_p=dropout_p)
        # Линейный слой для инициализации скрытого состояния LSTM
        self.init_h = nn.Linear(encoded_size, hidden_size)

    def forward(self, images, captions):
        """
        images  - входное изображение [B, C, H, W];
        captions - последовательность токенов (например, с включёнными спецтокенами <start> и <end>) [B, seq_len].
        Возвращает:
          logits для каждого временного шага [B, seq_len, vocab_size].
        """
        # Извлекаем признаки изображения с помощью энкодера
        features = self.encoder(images)  # [B, encoded_size]
        # Инициализируем скрытое состояние LSTM
        h0 = torch.tanh(self.init_h(features)).unsqueeze(0)  # [1, B, hidden_size]
        c0 = torch.zeros_like(h0)  # [1, B, hidden_size]
        hidden = (h0, c0)
        # Генерируем последовательность через декодер
        logits, hidden = self.decoder(captions, hidden)
        return logits


#################################
# Тестовый запуск модели
#################################
if __name__ == '__main__':
    # Задайте размер словаря в соответствии с вашим набором токенов
    vocab_size = 105
    # Создаем объект модели
    model = Im2LatexModel(vocab_size=vocab_size, encoded_size=512, embed_size=256, hidden_size=256, dropout_p=0.5)

    # Тестовый пример: создадим случайное изображение (размер 256x256, 3 канала, т.к. ResNet18 ожидает RGB)
    # Если ваши данные одноканальные, нужно повторить канал 3 раза.
    dummy_image = torch.randn(1, 3, 256, 256)

    # Пример последовательности токенов. Предположим, что:
    # индекс 1 используется для <start>, индекс 2 — для <end>, а остальные соответствуют символам.
    dummy_caption = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)  # [B, seq_len]

    # Пропускаем данные через модель
    output = model(dummy_image, dummy_caption)
    print("Output shape:", output.shape)  # Ожидаем: [1, seq_len, vocab_size]
