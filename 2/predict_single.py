import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

BASE_IMG_DIR = r"E:/3курс/курс/img_processed"
BASE_ANN_DIR = r"C:/Users/Admin/Desktop/KursProject"

from train_resnet_lstm import Im2LatexModel

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
vocab_list = special_tokens + base_vocab_list

token_to_idx = {token: i for i, token in enumerate(vocab_list)}
idx_to_token = {i: token for i, token in enumerate(vocab_list)}

def decode_tokens(token_ids, idx_to_token):
    tokens = []
    for tid in token_ids:
        token = idx_to_token.get(tid, "")
        if token == "<end>":
            break
        if token == "<start>":
            continue
        tokens.append(token)
    return " ".join(tokens)

def convert_to_rgb(img):
    return img.convert("RGB")

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(convert_to_rgb),
    transforms.ToTensor()
])

def beam_search_decode(model, image_tensor, start_token, beam_width=5, max_length=50, temperature=1.0, device="cpu"):
    model.eval()
    with torch.no_grad():
        features = model.encoder(image_tensor)
        h0 = torch.tanh(model.init_h(features)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)
    beam = [([start_token], 0.0, hidden)]
    for _ in range(max_length):
        new_beam = []
        for seq, score, hidden_state in beam:
            if seq[-1] == token_to_idx["<end>"]:
                new_beam.append((seq, score, hidden_state))
                continue
            input_token = torch.tensor([[seq[-1]]], device=device)
            logits, new_hidden = model.decoder(input_token, hidden_state)
            logits = logits[:, -1, :]
            logits = logits / temperature
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            sorted_vals, sorted_indices = torch.sort(log_probs, descending=True)
            for i in range(beam_width):
                next_token = sorted_indices[i].item()
                candidate_seq = seq + [next_token]
                candidate_score = score + sorted_vals[i].item()
                new_beam.append((candidate_seq, candidate_score, new_hidden))
        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
        beam = new_beam
        if all(candidate[0][-1] == token_to_idx["<end>"] for candidate in beam):
            break
    best_candidate = max(beam, key=lambda x: x[1])[0]
    return best_candidate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "ResNetCP_epoch_50.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

model = Im2LatexModel(vocab_size=len(vocab_list)).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Checkpoint loaded: {checkpoint_path} (Epoch {checkpoint['epoch']})")

image_path = os.path.join(BASE_IMG_DIR, r"train\CROHME2019\32_em_204.png")

image = Image.open(image_path)
image_tensor = test_transforms(image).unsqueeze(0).to(device)

predicted_indices = beam_search_decode(
    model,
    image_tensor,
    start_token=token_to_idx["<start>"],
    beam_width=5,
    max_length=50,
    temperature=1.0,
    device=device
)

predicted_expression = decode_tokens(predicted_indices, idx_to_token)
print("Predicted expression:")
print(predicted_expression)

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.title("Распознанное выражение:\n" + predicted_expression, fontsize=16)
plt.show()
