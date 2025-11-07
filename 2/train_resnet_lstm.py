import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset_crohme import MathExpressionDataset
from PIL import Image
import torchvision.models as models
from torchvision.models import ResNet18_Weights

BASE_IMG_DIR = r"E:/3курс/курс/img_processed"
BASE_ANN_DIR = r"C:/Users/Admin/Desktop/KursProject"

def convert_to_rgb(img):
    return img.convert("RGB")

special_tokens = ["<pad>", "<start>", "<end>"]

base_vocab_list = [
    '!', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', '=', 'A', 'B', 'C', 'COMMA', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P',
    'R', 'S', 'T', 'V', 'X', 'Y', '[', '\\Delta', '\\alpha', '\\beta', '\\cos', '\\div',
    '\\exists', '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int', '\\lambda',
    '\\ldots', '\\leq', '\\lim', '\\log', '\\lt', '\\mu', '\\neq', '\\phi', '\\pi', '\\pm',
    '\\prime', '\\rightarrow', '\\sigma', '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta',
    '\\times', '{', '}', ']', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}'
]

vocab_list = special_tokens + [t for t in base_vocab_list if t not in special_tokens]
token_to_idx = {token: i for i, token in enumerate(vocab_list)}
VOCAB_SIZE = len(vocab_list)
idx_to_token = {i: token for token, i in token_to_idx.items()}

def tokenize_caption(caption_dict):
    if "objects" not in caption_dict:
        raise ValueError("Caption dictionary must contain an 'objects' key.")
    toks = []
    for obj in caption_dict["objects"]:
        symbol = obj.get("symbol", "").strip()
        if symbol in {"\\{", "\\}"}:
            symbol = symbol[1:]
        if symbol == "":
            continue
        if symbol not in token_to_idx:
            raise ValueError(f"Token '{symbol}' not found in vocabulary!")
        toks.append(token_to_idx[symbol])
    token_sequence = [token_to_idx["<start>"]] + toks + [token_to_idx["<end>"]]
    return torch.tensor(token_sequence, dtype=torch.long)

def my_collate_fn(batch):
    images = []
    captions = []
    for item in batch:
        image, caption = item
        images.append(image)
        if isinstance(caption, dict):
            if "tokens" in caption:
                tokens = caption["tokens"]
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens, dtype=torch.long)
                if tokens[0].item() != token_to_idx["<start>"]:
                    tokens = torch.cat([torch.tensor([token_to_idx["<start>"]]), tokens])
                if tokens[-1].item() != token_to_idx["<end>"]:
                    tokens = torch.cat([tokens, torch.tensor([token_to_idx["<end>"]])])
                captions.append(tokens)
            elif "caption" in caption:
                tokens = caption["caption"]
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens, dtype=torch.long)
                if tokens[0].item() != token_to_idx["<start>"]:
                    tokens = torch.cat([torch.tensor([token_to_idx["<start>"]]), tokens])
                if tokens[-1].item() != token_to_idx["<end>"]:
                    tokens = torch.cat([tokens, torch.tensor([token_to_idx["<end>"]])])
                captions.append(tokens)
            else:
                tokens = tokenize_caption(caption)
                captions.append(tokens)
        elif isinstance(caption, torch.Tensor):
            captions.append(caption)
        else:
            captions.append(torch.tensor(caption, dtype=torch.long))
    images = torch.stack(images, dim=0)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=token_to_idx["<pad>"])
    return images, captions_padded

class ResNetEncoder(nn.Module):
    def __init__(self, encoded_size=512, pretrained=True):
        super(ResNetEncoder, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, encoded_size)

    def forward(self, x):
        features = self.features(x)
        pooled = self.adaptive_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        return self.fc(pooled)

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=256, dropout_p=0.6):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, captions, hidden):
        embeds = self.embedding(captions)
        embeds = self.dropout(embeds)
        outputs, hidden = self.lstm(embeds, hidden)
        outputs = self.dropout(outputs)
        return self.fc(outputs), hidden

    def step(self, input_token, hidden):
        embed = self.embedding(input_token)
        out, hidden = self.lstm(embed, hidden)
        logits = self.fc(out)
        return logits, hidden

class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, encoded_size=512, embed_size=256, hidden_size=256, dropout_p=0.6, pretrained_encoder=True):
        super(Im2LatexModel, self).__init__()
        self.encoder = ResNetEncoder(encoded_size=encoded_size, pretrained=pretrained_encoder)
        self.decoder = LSTMDecoder(vocab_size, embed_size, hidden_size, dropout_p=dropout_p)
        self.init_h = nn.Linear(encoded_size, hidden_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        h0 = torch.tanh(self.init_h(features)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)
        logits, hidden = self.decoder(captions, hidden)
        return logits

def temperature_decode(model, image, start_token, temperature=1.0, max_length=50, device="cpu"):
    model.eval()
    with torch.no_grad():
        features = model.encoder(image)
        h0 = torch.tanh(model.init_h(features)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)
        input_token = torch.tensor([[start_token]], device=device)
        output_tokens = []
        for _ in range(max_length):
            logits, hidden = model.decoder.step(input_token, hidden)
            logits = logits[:, -1, :]
            scaled_logits = logits / max(temperature, 1e-8)
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            output_tokens.append(next_token)
            if next_token == token_to_idx["<end>"]:
                break
            input_token = torch.tensor([[next_token]], device=device)
        return output_tokens

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_dict = checkpoint.get('model_state_dict', checkpoint)
    model_dict = model.state_dict()
    filtered_dict = {}
    for k, v in checkpoint_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
        else:
            print(f"Пропуск ключа {k}: сохраненная форма {v.shape}, текущая форма {model_dict[k].shape if k in model_dict else 'нет'}")
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    return checkpoint.get('epoch', 0)

def main():
    train_img_folder = os.path.join(BASE_IMG_DIR, "train")
    val_img_folder = os.path.join(BASE_IMG_DIR, "val")
    base_annotations_path = BASE_ANN_DIR
    train_annotation_file = os.path.join(base_annotations_path, "train.json")
    val_annotation_file = os.path.join(base_annotations_path, "val.json")

    train_data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(convert_to_rgb),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor()
    ])

    train_dataset = MathExpressionDataset(train_img_folder, train_annotation_file, train_data_transforms)
    val_dataset = MathExpressionDataset(val_img_folder, val_annotation_file, val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                            collate_fn=my_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Im2LatexModel(vocab_size=VOCAB_SIZE, encoded_size=512, embed_size=256, hidden_size=256, dropout_p=0.6, pretrained_encoder=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_idx["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    total_epochs = 50
    checkpoint_path_30 = "ResNetCP_epoch_30.pth"
    if os.path.exists(checkpoint_path_30):
        print("Checkpoint for epoch 30 найден. Загружаем его и возобновляем обучение.")
        start_epoch = load_checkpoint(model, checkpoint_path_30, device)
        print(f"Начинаем обучение с эпохи {start_epoch + 1}.")
    else:
        start_epoch = 0
        print("Чекпоинт для эпохи 30 не найден. Обучение начинается с нуля.")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            if captions.size(1) < 2:
                continue
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]
            outputs = model(images, input_captions)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), target_captions.contiguous().view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch + 1}/{total_epochs}], Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                if captions.size(1) < 2:
                    continue
                input_captions = captions[:, :-1]
                target_captions = captions[:, 1:]
                outputs = model(images, input_captions)
                loss = criterion(outputs.view(-1, VOCAB_SIZE), target_captions.contiguous().view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch [{epoch + 1}/{total_epochs}], Validation Loss: {avg_val_loss:.4f}")

        temperature = 0.8
        print("Примеры генерации (temperature sampling) на валидационном наборе:")
        for idx in range(min(3, len(val_dataset))):
            image, _ = val_dataset[idx]
            image = image.unsqueeze(0).to(device)
            pred_tokens = temperature_decode(model, image, start_token=token_to_idx["<start>"],
                                             temperature=temperature, max_length=50, device=device)
            pred_caption = " ".join([idx_to_token.get(t, "<UNK>") for t in pred_tokens])
            print(f"Example {idx + 1}: {pred_caption}")

        checkpoint_path = f"ResNetCP_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    main()
