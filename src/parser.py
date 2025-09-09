import pandas as pd
from torch.utils.data import Dataset

class DonutDataset(Dataset):
    """
    Custom dataset for the Donut model.
    """
    def __init__(self, csv_path, processor):
        self.df = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the image and text from the dataframe
        image = self.df.loc[idx, "image"]
        text = self.df.loc[idx, "text"]

        # Process the image and text
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text, return_tensors="pt").input_ids

        return {"pixel_values": pixel_values.squeeze(), "labels": labels.squeeze()}

def get_dataset(csv_path, processor):
    """
    Returns a DonutDataset object.
    """
    return DonutDataset(csv_path, processor)
