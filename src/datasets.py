from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import os


class DonutCsvDataset(Dataset):
    """
    Expect CSV with columns: image_path, text
    """

    def __init__(self, csv_path, processor, image_root="."):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.image_root = image_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        text = str(row["text"])
        labels = self.processor.tokenizer(
            text, add_special_tokens=True, return_tensors="pt"
        ).input_ids.squeeze(0)

        # Trainer uses -100 as ignore_index
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


class DonutDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        # labels are variable-length → pad
        labels = [b["labels"] for b in batch]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        labels_padded[labels_padded == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels_padded}

