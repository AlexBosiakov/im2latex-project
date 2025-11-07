import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_crohme import MathExpressionDataset
from train_resnet_lstm import Im2LatexModel
import torch.nn.functional as F

BASE_IMG_DIR = r"E:/3курс/курс/img_processed"
BASE_ANN_DIR = r"C:/Users/Admin/Desktop/KursProject"

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
vocab_list = list(dict.fromkeys(vocab_list))
token_to_idx = {t: i for i, t in enumerate(vocab_list)}
idx_to_token = {i: t for i, t in enumerate(vocab_list)}

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
        try:
            idx = vocab_list.index(symbol)
        except ValueError:
            raise ValueError(f"Token '{symbol}' not found in vocabulary!")
        toks.append(idx)
    token_sequence = [token_to_idx["<start>"]] + toks + [token_to_idx["<end>"]]
    return torch.tensor(token_sequence, dtype=torch.long)

def my_collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    images = []
    captions = []
    for image, caption in batch:
        images.append(image)
        if isinstance(caption, dict):
            tokens = tokenize_caption(caption)
            captions.append(tokens)
        elif isinstance(caption, torch.Tensor):
            captions.append(caption)
        else:
            captions.append(torch.tensor(caption, dtype=torch.long))
    images = torch.stack(images, dim=0)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=token_to_idx["<pad>"])
    return images, captions_padded

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
            logits = logits / max(temperature, 1e-8)
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
            topk_log_probs = topk_log_probs.squeeze(0)
            topk_indices = topk_indices.squeeze(0)
            for i in range(beam_width):
                next_token = topk_indices[i].item()
                candidate_seq = seq + [next_token]
                candidate_score = score + topk_log_probs[i].item()
                new_beam.append((candidate_seq, candidate_score, new_hidden))
        new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
        beam = new_beam
        if all(candidate[0][-1] == token_to_idx["<end>"] for candidate in beam):
            break
    best_candidate = max(beam, key=lambda x: x[1])[0]
    return best_candidate

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[m][n]

def normalized_similarity(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)

test_img_folder = os.path.join(BASE_IMG_DIR, "test")
test_annotation_file = os.path.join(BASE_ANN_DIR, "test.json")

test_dataset = MathExpressionDataset(test_img_folder, test_annotation_file, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Im2LatexModel(vocab_size=len(vocab_list)).to(device)

checkpoint_path = "ResNetCP_epoch_50.pth"
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
print(f"Checkpoint loaded: {checkpoint_path} (Epoch {checkpoint.get('epoch', 'unknown')})")

model.eval()
total_samples = 0
exact_match_count = 0
similarity_sum = 0.0
acceptable_threshold = 0.8
acceptable_count = 0

with torch.no_grad():
    for images, captions in test_loader:
        images = images.to(device)
        gt_tokens = captions[0].tolist()
        gt_caption = decode_tokens(gt_tokens, idx_to_token).strip()
        predicted_indices = beam_search_decode(
            model,
            images,
            start_token=token_to_idx["<start>"],
            beam_width=5,
            max_length=50,
            temperature=1.0,
            device=device
        )
        predicted_caption = decode_tokens(predicted_indices, idx_to_token).strip()
        total_samples += 1
        if predicted_caption == gt_caption:
            exact_match_count += 1
        sim = normalized_similarity(gt_caption, predicted_caption)
        similarity_sum += sim
        if sim >= acceptable_threshold:
            acceptable_count += 1

exact_match_accuracy = (exact_match_count / total_samples) * 100 if total_samples > 0 else 0.0
average_similarity = (similarity_sum / total_samples) * 100 if total_samples > 0 else 0.0
acceptable_percentage = (acceptable_count / total_samples) * 100 if total_samples > 0 else 0.0

print(f"\nТестовая выборка: {total_samples} примеров")
print(f"Точный матч: {exact_match_accuracy:.2f}%")
print(f"Средняя схожесть (soft metric): {average_similarity:.2f}%")
print(f"Процент примеров с схожестью >= {acceptable_threshold*100:.0f}%: {acceptable_percentage:.2f}%")
