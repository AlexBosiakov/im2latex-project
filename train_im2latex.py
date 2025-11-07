import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from Dataset import MathExpressionDataset
from Im2LatexModel import Im2LatexModel

BASE_IMG_DIR = r"E:/3курс/курс/img_processed"  # пример базовой папки для изображений
BASE_ANN_DIR = r"C:/Users/Admin/Desktop/KursProject"  # пример базовой папки для аннотаций

train_img_folder = os.path.join(BASE_IMG_DIR, "train")
val_img_folder = os.path.join(BASE_IMG_DIR, "val")

train_annotation_file = os.path.join(BASE_ANN_DIR, "train.json")
val_annotation_file = os.path.join(BASE_ANN_DIR, "val.json")

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
token_to_idx = {token: i for i, token in enumerate(vocab_list)}


def tokenize_caption(caption_dict):
    if "objects" not in caption_dict:
        raise ValueError("Caption dictionary must contain an 'objects' key.")
    tokens = []
    for obj in caption_dict["objects"]:
        symbol = obj.get("symbol", "").strip()
        if symbol in {"\\{", "\\}"}:
            symbol = symbol[1:]
        if symbol == "":
            continue
        if symbol not in token_to_idx:
            raise ValueError(f"Token '{symbol}' not found in vocabulary!")
        tokens.append(token_to_idx[symbol])
    return torch.tensor(tokens, dtype=torch.long)


def my_collate_fn(batch):
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
        else:
            processed_captions.append(torch.tensor(caption, dtype=torch.long))

    captions_padded = pad_sequence(processed_captions, batch_first=True, padding_value=0)
    return images, captions_padded


def main():
    # Преобразования данных
    train_data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = MathExpressionDataset(train_img_folder, train_annotation_file, train_data_transforms)
    val_dataset = MathExpressionDataset(val_img_folder, val_annotation_file, val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=my_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                            collate_fn=my_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VOCAB_SIZE = len(vocab_list)
    model = Im2LatexModel(vocab_size=VOCAB_SIZE, hidden_size=256).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    resume_checkpoint = "im2latex_checkpoint_epoch_25.pth"
    start_epoch = 25

    if os.path.exists(resume_checkpoint):
        print(f"Resuming training from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        print("Starting training from scratch.")
        start_epoch = 0

    num_epochs = 30

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]

            outputs = model(images, input_captions)
            outputs = outputs.reshape(-1, VOCAB_SIZE)
            target_captions = target_captions.reshape(-1)

            loss = criterion(outputs, target_captions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                input_captions = captions[:, :-1]
                target_captions = captions[:, 1:]

                outputs = model(images, input_captions)
                outputs = outputs.reshape(-1, VOCAB_SIZE)
                target_captions = target_captions.reshape(-1)

                loss = criterion(outputs, target_captions)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        checkpoint_path = f"im2latex_checkpoint_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

        print(f"Checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    main()
