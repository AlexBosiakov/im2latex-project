import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from dataset_crohme import MathExpressionDataset
from train_resnet_lstm import Im2LatexModel

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
vocab_list = special_tokens + [t for t in base_vocab_list if t not in special_tokens]
vocab_list = list(dict.fromkeys(vocab_list))
token_to_idx = {t: i for i, t in enumerate(vocab_list)}
idx_to_token = {i: t for i, t in enumerate(vocab_list)}

def convert_to_rgb(img):
    return img.convert("RGB")

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(convert_to_rgb),
    transforms.ToTensor()
])

def decode_tokens(token_ids, idx_to_token):
    tokens = []
    for tid in token_ids:
        t = idx_to_token.get(tid, "")
        if t == "<end>":
            break
        if t == "<start>":
            continue
        tokens.append(t)
    return " ".join(tokens)

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i, c1 in enumerate(s1, 1):
        cur = [i] + [0] * n
        for j, c2 in enumerate(s2, 1):
            add = prev[j] + 1
            delete = cur[j - 1] + 1
            replace = prev[j - 1] + (0 if c1 == c2 else 1)
            cur[j] = min(add, delete, replace)
        prev = cur
    return prev[-1]

def normalized_similarity(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    d = levenshtein_distance(s1, s2)
    m = max(len(s1), len(s2))
    return 1.0 - (d / m)

def caption_dict_to_string(caption_dict):
    if not isinstance(caption_dict, dict):
        return str(caption_dict).strip()
    if "objects" not in caption_dict:
        return str(caption_dict).strip()
    symbols = [obj.get("symbol", "").strip() for obj in caption_dict["objects"] if obj.get("symbol", "").strip() != ""]
    return " ".join(symbols)

def beam_search_decode(model, image_tensor, start_token, beam_width=5, max_length=50, temperature=1.0, device="cpu"):
    model.eval()
    with torch.no_grad():
        features = model.encoder(image_tensor)
        h0 = torch.tanh(model.init_h(features)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)
    beam = [([start_token], 0.0, hidden)]
    for _ in range(max_length):
        candidates = []
        for seq, score, hidden_state in beam:
            if seq[-1] == token_to_idx["<end>"]:
                candidates.append((seq, score, hidden_state))
                continue
            input_token = torch.tensor([[seq[-1]]], device=device)
            logits, new_hidden = model.decoder(input_token, hidden_state)
            logits = logits[:, -1, :]
            logits = logits / max(temperature, 1e-8)
            logp = torch.log_softmax(logits, dim=-1).squeeze(0)
            topk = torch.topk(logp, k=min(beam_width, logp.numel()))
            for v, idx in zip(topk.values.cpu().numpy(), topk.indices.cpu().numpy()):
                candidates.append((seq + [int(idx)], score + float(v), new_hidden))
        if not candidates:
            break
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        beam = candidates
        if all(seq[-1] == token_to_idx["<end>"] for seq, _, _ in beam):
            break
    best = max(beam, key=lambda x: x[1])[0]
    return best

test_img_folder = os.path.join(BASE_IMG_DIR, "test")
test_annotation_file = os.path.join(BASE_ANN_DIR, "test.json")

test_dataset = MathExpressionDataset(test_img_folder, test_annotation_file, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Im2LatexModel(vocab_size=len(vocab_list)).to(device)

checkpoint_path = "ResNetCP_epoch_50.pth"
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
print(f"Checkpoint loaded: {checkpoint_path} (Epoch {checkpoint.get('epoch', 'unknown')})")

results = []
for idx in tqdm(range(len(test_dataset)), desc="Computing soft metrics"):
    image, caption = test_dataset[idx]
    file_key = test_dataset.file_names[idx] if hasattr(test_dataset, "file_names") else f"idx_{idx}"
    if not torch.is_tensor(image):
        image_tensor = test_transforms(image).unsqueeze(0).to(device)
    else:
        image_tensor = image.unsqueeze(0).to(device)
    predicted_indices = beam_search_decode(
        model,
        image_tensor,
        start_token=token_to_idx["<start>"],
        beam_width=5,
        max_length=50,
        temperature=1.0,
        device=device
    )
    predicted_caption = decode_tokens(predicted_indices, idx_to_token).strip()
    if isinstance(caption, dict):
        gt_caption = caption_dict_to_string(caption).strip()
    elif torch.is_tensor(caption):
        gt_caption = decode_tokens(caption.tolist(), idx_to_token).strip()
    else:
        gt_caption = str(caption).strip()
    sim = normalized_similarity(gt_caption, predicted_caption)
    results.append((file_key, gt_caption, predicted_caption, sim))

results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
top200 = results_sorted[:200]

print("\n---- Top 200 Examples (Highest Soft Metric) ----")
for file_key, gt_caption, predicted_caption, sim in top200:
    print(f"File: {file_key}")
    print(f"Ground Truth: {gt_caption}")
    print(f"Predicted:    {predicted_caption}")
    print(f"Soft Metric:  {sim * 100:.2f}%")
    print("---------------------------")
