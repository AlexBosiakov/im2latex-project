import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. CNN-энкодер
class EncoderCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, dropout_p=0.5):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x должен быть уже на нужном устройстве
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        # Переставляем оси: [batch, channels, height, width] -> [batch, width, channels*height]
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, width, channels * height)
        return x


# 2. RNN-декодер
class DecoderRNN(nn.Module):
    def __init__(self, feature_dim, hidden_size, vocab_size, num_layers=1, dropout_p=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # Инициализация скрытого состояния на основе признаков изображения
        self.init_h = nn.Linear(feature_dim, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_p if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # captions: [batch, seq_len]
        embeddings = self.embedding(captions)  # [batch, seq_len, hidden_size]
        embeddings = self.dropout(embeddings)
        # Вычисляем среднее по временной оси (т.е. по ширине, как после энкодера)
        mean_features = features.mean(dim=1)  # [batch, feature_dim]
        h0 = torch.tanh(self.init_h(mean_features))  # [batch, hidden_size]
        h0 = h0.unsqueeze(0)  # [1, batch, hidden_size]
        c0 = torch.zeros_like(h0)  # [1, batch, hidden_size]
        outputs, _ = self.lstm(embeddings, (h0, c0))  # [batch, seq_len, hidden_size]
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)  # [batch, seq_len, vocab_size]
        return outputs


# 3. Главная модель Im2Latex с объединением энкодера и декодера
class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, dropout_p=0.5):
        super(Im2LatexModel, self).__init__()
        self.encoder = EncoderCNN(in_channels=1, out_channels=256, dropout_p=dropout_p)
        # Для изображений размером 256x256, после трёх пула получается ширина = 32: feature_dim = 256 * 32 = 8192.
        self.decoder = DecoderRNN(feature_dim=256 * 32, hidden_size=hidden_size, vocab_size=vocab_size,
                                  dropout_p=dropout_p)

    def forward(self, images, captions):
        # Предполагается, что images и captions уже перенесены на нужное устройство (например, GPU)
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


# Пример простого тестового запуска
if __name__ == "__main__":
    # Создаем экземпляр модели, передаем количество токенов (например, 105) и желаемый dropout (0.5)
    model = Im2LatexModel(vocab_size=105, hidden_size=256, dropout_p=0.5)
    model.to(device)

    # Создаем фиктивные входы: 4 изображения размером 256x256 (одноканальные) и случайные подписи длиной 20
    dummy_images = torch.randn(4, 1, 256, 256).to(device)
    dummy_captions = torch.randint(0, 105, (4, 20)).to(device)

    outputs = model(dummy_images, dummy_captions)
    print("Output shape:", outputs.shape)
