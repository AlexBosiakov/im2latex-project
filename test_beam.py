import os
import json
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from Im2LatexModel import Im2LatexModel
from Dataset import MathExpressionDataset

BASE_IMG_DIR = r"E:/3курс/курс/img_processed"
BASE_ANN_DIR = r"C:/Users/Admin/Desktop/KursProject"

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
token_to_idx = {token: i for i, token in enumerate(vocab_list)}
idx_to_token = {i: token for token, i in token_to_idx.items()}


def beam_search_decode(model, image, device, beam_width=3, max_length=50, start_token=0, end_token=None):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        features = model.encoder(image)
        mean_features = features.mean(dim=1)
        h0 = torch.tanh(model.decoder.init_h(mean_features)).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        beam = [([start_token], h0, c0, 0.0)]

        for _ in range(max_length):
            new_beam = []
            for seq, h, c, cum_log_prob in beam:
                if end_token is not None and seq[-1] == end_token:
                    new_beam.append((seq, h, c, cum_log_prob))
                    continue

                token_input = torch.tensor([[seq[-1]]], dtype=torch.long).to(device)
                embedding = model.decoder.embedding(token_input)
                output, (new_h, new_c) = model.decoder.lstm(embedding, (h, c))
                logits = model.decoder.fc(output.squeeze(1))
                log_probs = F.log_softmax(logits, dim=1)

                top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
                top_log_probs = top_log_probs.squeeze(0)
                top_indices = top_indices.squeeze(0)

                for log_prob, token_idx in zip(top_log_probs, top_indices):
                    new_seq = seq + [token_idx.item()]
                    new_cum_log_prob = cum_log_prob + log_prob.item()
                    new_beam.append((new_seq, new_h, new_c, new_cum_log_prob))

            beam = sorted(new_beam, key=lambda x: x[3], reverse=True)[:beam_width]
            if end_token is not None and all(seq[-1] == end_token for seq, _, _, _ in beam):
                break

        best_seq = sorted(beam, key=lambda x: x[3], reverse=True)[0][0]
        return best_seq[1:]


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalized_similarity(s1, s2):
    if not s1 and not s2:
        return 1.0
    lev_distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    similarity = 1 - (lev_distance / max_len)
    return similarity


def process_ground_truth(caption):
    if isinstance(caption, dict):
        if "caption" in caption:
            return str(caption["caption"]).strip()
        elif "objects" in caption:
            tokens = []
            for obj in caption["objects"]:
                if isinstance(obj, dict):
                    symbol = obj.get("symbol", "")
                    if isinstance(symbol, str):
                        tokens.append(symbol.strip())
                    else:
                        tokens.append(str(symbol))
                elif isinstance(obj, list):
                    tokens.append(" ".join([str(item).strip() for item in obj]))
                elif isinstance(obj, str):
                    tokens.append(obj.strip())
                else:
                    tokens.append(str(obj).strip())
            return " ".join(tokens)
        else:
            return str(caption).strip()
    elif isinstance(caption, list):
        tokens = []
        for elem in caption:
            if isinstance(elem, dict):
                if "symbol" in elem:
                    sym = elem["symbol"]
                    if isinstance(sym, str):
                        tokens.append(sym.strip())
                    else:
                        tokens.append(str(sym))
                else:
                    tokens.append(str(elem).strip())
            elif isinstance(elem, str):
                tokens.append(elem.strip())
            else:
                tokens.append(str(elem).strip())
        return " ".join(tokens)
    else:
        return str(caption).strip()


def test_model_with_beam_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_img_folder = os.path.join(BASE_IMG_DIR, "test")
    test_annotation_file = os.path.join(BASE_ANN_DIR, "test.json")

    test_dataset = MathExpressionDataset(test_img_folder, test_annotation_file, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    VOCAB_SIZE = len(vocab_list)
    model = Im2LatexModel(vocab_size=VOCAB_SIZE, hidden_size=256).to(device)

    checkpoint_path = "im2latex_checkpoint_epoch_30.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Загружен чекпоинт: {checkpoint_path}")

    total_similarity = 0.0
    sample_count = 0

    for images, captions in test_loader:
        images = images.to(device)
        image = images[0]

        generated_token_seq = beam_search_decode(model, image, device,
                                                 beam_width=3, max_length=50,
                                                 start_token=0, end_token=None)
        generated_caption = " ".join([idx_to_token[token] for token in generated_token_seq])
        gt_caption = process_ground_truth(captions)

        sim = normalized_similarity(generated_caption, gt_caption)
        total_similarity += sim
        sample_count += 1

    avg_similarity = total_similarity / sample_count if sample_count > 0 else 0
    print(f"\nСредняя схожесть (Soft metric) по тестовому набору: {avg_similarity * 100:.2f}%")

if __name__ == "__main__":
    test_model_with_beam_search()
