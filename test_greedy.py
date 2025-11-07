import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import Levenshtein

BASE_IMG_DIR = r"E:\3курс\курс\img_processed"
CHECKPOINT_PATH = "im2latex_checkpoint_epoch_30.pth"

class EncoderCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, dropout_p=0.5):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, width, channels * height)
        return x

class DecoderRNN(nn.Module):
    def __init__(self, feature_dim, hidden_size, vocab_size, num_layers=1, dropout_p=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.init_h = nn.Linear(feature_dim, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_p if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = self.dropout(embeddings)
        mean_features = features.mean(dim=1)
        h0 = torch.tanh(self.init_h(mean_features)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        outputs, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)
        return outputs

class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, dropout_p=0.5):
        super(Im2LatexModel, self).__init__()
        self.encoder = EncoderCNN(in_channels=1, out_channels=256, dropout_p=dropout_p)
        self.decoder = DecoderRNN(feature_dim=256 * 32, hidden_size=hidden_size, vocab_size=vocab_size,
                                  dropout_p=dropout_p)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

special_tokens = ["<pad>", "<start>", "<end>"]
base_vocab_list = [
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w',
    'x','y','z',
    '0','1','2','3','4','5','6','7','8','9',
    '+','-','=','(',')','[',']','{','}','\\','/ ',',','.',';'
]
vocab_list = special_tokens + base_vocab_list
vocab_list = list(dict.fromkeys(vocab_list))
token_to_idx = {token: i for i, token in enumerate(vocab_list)}
idx_to_token = {i: token for token, i in token_to_idx.items()}
START_TOKEN = token_to_idx["<start>"]
END_TOKEN = token_to_idx["<end>"]
VOCAB_SIZE = len(vocab_list)

def generate_caption(model, image, start_token, max_length=50, device="cpu"):
    model.eval()
    with torch.no_grad():
        features = model.encoder(image)
        mean_features = features.mean(dim=1)
        h0 = torch.tanh(model.decoder.init_h(mean_features)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)
        input_token = torch.tensor([[start_token]], device=device)
        generated_indices = []
        for _ in range(max_length):
            embedding = model.decoder.embedding(input_token)
            output, hidden = model.decoder.lstm(embedding, hidden)
            output = model.decoder.fc(output.squeeze(1))
            next_token = output.argmax(dim=1).item()
            generated_indices.append(next_token)
            if next_token == END_TOKEN:
                break
            input_token = torch.tensor([[next_token]], device=device)
        return generated_indices

def indices_to_string(indices):
    tokens = [idx_to_token.get(idx, "<UNK>") for idx in indices]
    filtered = [t for t in tokens if t not in ["<start>", "<end>", "<pad>"]]
    return " ".join(filtered)

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    filtered = {}
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            print(f"Skipping key {k}: checkpoint shape {v.shape} vs model {model_state.get(k).shape if k in model_state else 'MISSING'}")
    model_state.update(filtered)
    model.load_state_dict(model_state)
    epoch = checkpoint.get("epoch", "unknown")
    print(f"Checkpoint loaded from {checkpoint_path}, epoch={epoch}")

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_image_rel = os.path.join("test", "CROHME2019_test", "ISICal19_1201_em_750.png")
    test_image_path = os.path.join(BASE_IMG_DIR, test_image_rel)

    if not os.path.isfile(test_image_path):
        raise FileNotFoundError(f"Test image not found: {test_image_path}")

    model = Im2LatexModel(vocab_size=VOCAB_SIZE, hidden_size=256, dropout_p=0.5)
    model = model.to(device)

    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    load_checkpoint(model, CHECKPOINT_PATH, device)

    image = Image.open(test_image_path).convert("RGB")
    image = test_transforms(image).unsqueeze(0).to(device)

    generated_indices = generate_caption(model, image, START_TOKEN, max_length=50, device=device)
    generated_caption = indices_to_string(generated_indices)
    print("Generated Caption:", generated_caption)

    ground_truth_caption = "a + b = c"
    similarity = Levenshtein.ratio(generated_caption, ground_truth_caption)
    print("Soft Metric (Normalized Levenshtein similarity):", similarity)
