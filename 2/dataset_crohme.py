import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE_IMG_DIR = r"E:/3курс/курс/img_processed"
BASE_ANN_DIR = r"C:/Users/Admin/Desktop/KursProject"

class MathExpressionDataset(Dataset):
    def __init__(self, img_folder, annotation_file, transform=None):
        self.img_folder = img_folder
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        self.file_names = [key for key in self.annotations.keys() if not key.startswith("~$")]
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        key = self.file_names[idx]
        img_file = os.path.splitext(key)[0] + ".png"

        found_path = None
        for root, _, files in os.walk(self.img_folder):
            if img_file in files:
                found_path = os.path.join(root, img_file)
                break

        if found_path is None:
            raise FileNotFoundError(f"Файл не найден: {img_file} в {self.img_folder}")

        image = Image.open(found_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        caption = self.annotations[key]
        return image, caption


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_img_folder = os.path.join(BASE_IMG_DIR, "train")
    val_img_folder = os.path.join(BASE_IMG_DIR, "val")
    test_img_folder = os.path.join(BASE_IMG_DIR, "test")

    train_annotation_file = os.path.join(BASE_ANN_DIR, "train.json")
    val_annotation_file = os.path.join(BASE_ANN_DIR, "val.json")
    test_annotation_file = os.path.join(BASE_ANN_DIR, "test.json")

    train_dataset = MathExpressionDataset(train_img_folder, train_annotation_file, data_transforms)
    val_dataset = MathExpressionDataset(val_img_folder, val_annotation_file, data_transforms)
    test_dataset = MathExpressionDataset(test_img_folder, test_annotation_file, data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_idx, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        print(f"Batch {batch_idx}: images shape = {images.shape}")
        print(f"Sample caption: {captions[0]}")
        break
