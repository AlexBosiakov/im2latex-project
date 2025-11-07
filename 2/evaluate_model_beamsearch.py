import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset_crohme import MathExpressionDataset

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
VOCAB_SIZE = len(vocab_list)
token_to_idx = {token: i for i, token in enumerate(vocab_list)}
idx_to_token = {i: token for i, token in enumerate(vocab_list)}

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

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

def custom_collate_fn(batch):
    images = []
    captions_list = []
    for item in batch:
        images.append(item[0])
        cap = item[1]
        if isinstance(cap, dict):
            cap = tokenize_caption(cap)
        elif not isinstance(cap, torch.Tensor):
            cap = torch.tensor(cap, dtype=torch.long)
        captions_list.append(cap)
    images = torch.stack(images, dim=0)
    captions_padded = nn.utils.rnn.pad_sequence(captions_list, batch_first=True, padding_value=token_to_idx["<pad>"])
    return images, captions_padded

def beam_search_decode(model, image_tensor, start_token, beam_width=5, max_length=50, temperature=1.0, device="cpu"):
    model.eval()
    with torch.no_grad():
        features = model.encoder(image_tensor)
        if hasattr(model, 'init_h'):
            h0 = torch.tanh(model.init_h(features)).unsqueeze(0)
        elif hasattr(model.decoder, 'init_h'):
            h0 = torch.tanh(model.decoder.init_h(features)).unsqueeze(0)
        else:
            raise AttributeError("Нет слоя init_h в модели или в декодере!")
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
            logits, new_hidden = model.decoder.forward(input_token, hidden_state)
            logits = logits[:, -1, :]
            logits = logits / temperature
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

test_img_folder = os.path.join(BASE_IMG_DIR, "test")
test_annotation_file = os.path.join(BASE_ANN_DIR, "test.json")

test_dataset = MathExpressionDataset(test_img_folder, test_annotation_file, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

from Im2LatexModel import Im2LatexModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Im2LatexModel(vocab_size=108, hidden_size=256, dropout_p=0.5).to(device)

checkpoint_path = "ResNetCP_epoch_50.pth"
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
print(f"Checkpoint loaded: {checkpoint_path} (Epoch {checkpoint['epoch']})")

model.eval()
total_samples = 0
exact_match_count = 0
similarity_sum = 0.0
acceptable_threshold = 0.8
acceptable_count = 0
sample_examples = 0
max_examples_to_show = 5

with torch.no_grad():
    for idx, (images, captions) in enumerate(test_loader):
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

        if sample_examples < max_examples_to_show:
            print(f"\nExample {sample_examples + 1}:")
            print(f"Ground truth : {gt_caption}")
            print(f"Prediction   : {predicted_caption}")
            print(f"Similarity   : {sim * 100:.2f}%")
            sample_examples += 1

exact_match_accuracy = (exact_match_count / total_samples) * 100 if total_samples > 0 else 0.0
average_similarity = (similarity_sum / total_samples) * 100 if total_samples > 0 else 0.0
acceptable_percentage = (acceptable_count / total_samples) * 100 if total_samples > 0 else 0.0

print(f"\nTotal samples: {total_samples}")
print(f"Exact match  : {exact_match_accuracy:.2f}%")
print(f"Average similarity (soft metric): {average_similarity:.2f}%")
print(f"Examples with similarity >= {acceptable_threshold * 100:.0f}%: {acceptable_percentage:.2f}%")
