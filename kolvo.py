import json
import os


def collect_tokens_from_json(json_file_path):

    tokens = set()
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Проходим по каждому выражению (например, "001-equation000.lg")
        for expr_id, expr_data in data.items():
            objects = expr_data.get("objects", [])
            for obj in objects:
                symbol = obj.get("symbol", "").strip()
                if symbol:
                    tokens.add(symbol)
    return tokens


def collect_tokens_from_directory(directory_path):

    all_tokens = set()
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(directory_path, file_name)
            tokens = collect_tokens_from_json(file_path)
            all_tokens.update(tokens)
    return all_tokens


if __name__ == "__main__":
    json_folder = r"C:\Users\Admin\Desktop\KursProject"
    unique_tokens = collect_tokens_from_directory(json_folder)



    special_tokens = {"^", "_", "{", "}"}
    unique_tokens.update(special_tokens)

    print("Number of unique tokens:", len(unique_tokens))
    print("Unique tokens:")
    for token in sorted(unique_tokens):
        print(token)
