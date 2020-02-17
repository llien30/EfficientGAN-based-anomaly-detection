from torch.utils import data
from PIL import Image
import pandas as pd


class Dataset(data.Dataset):
    def __init__(self, csv_file, transform="None"):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["img_path"]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        cls_id = self.df.iloc[idx]["cls_id"]
        cls_label = self.df.iloc[idx]["cls_label"]

        sample = {
            "img": img,
            "cls_id": cls_id,
            "label": cls_label,
            "img_path": img_path,
        }

        return sample
