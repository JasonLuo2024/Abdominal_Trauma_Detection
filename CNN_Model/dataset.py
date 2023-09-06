__author__ = "JasonLuo"
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
train_csv = r'C:\Users\a6649\Desktop\Honors\Dataset\train.csv'

def getLabel(patient_ID):
    label = []
    df = pd.read_csv(train_csv, delimiter=',')
    filtered_df = df[df['patient_id'] == int(patient_ID)]
    if not filtered_df.empty:
        for catogry in ['extravasation_healthy','bowel_healthy','kidney_healthy','liver_healthy','spleen_healthy']:
            label.append(int(filtered_df.iloc[0][catogry]))
        return label



class PNGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_list = []
        self.label_list = []

        for root, directories, files in os.walk(root_dir):
            patient_ID = os.path.basename(root)
            try:
                label = getLabel(patient_ID)
                for file in files:
                    file_path = os.path.join(root, file)
                    self.image_list.append(file_path)
                    self.label_list.append(label)
            except Exception as e:
                continue


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img = Image.open(self.image_list[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.label_list[idx],dtype=torch.long)
        return img,label


def list_extend(y_true,y_pred,labels,preds):
    for index, value in enumerate(labels):
        y_true[index].extend(value)
    for index, value in enumerate(preds):
        y_pred[index].extend(value)

