__author__ = "JasonLuo"
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torchvision.models as models
from tqdm.auto import tqdm
import torch.nn as nn
import pandas as pd
from dataset import PNGDataset


# the purpose of iteration 1 is to experiment how accurate the model can detect liver trauma.
# all the dataset should be classified to either traumatic or non-traumatic.
# if the model works fine, we can apply the same model to the reset area.
# this model is based on multi_Channel_CNN
#

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

avg_accuracy = 0
avg_sensitivity = 0
avg_F1_score = 0


class MultiDensennet169(nn.Module):
    def __init__(self):
        super(MultiDensennet169, self).__init__()
        self.extravasation_model = models.densenet169(pretrained=True)
        self.bowel_model = models.densenet169(pretrained=True)
        self.kidney_model = models.densenet169(pretrained=True)
        self.liver_model = models.densenet169(pretrained=True)
        self.spleen_model = models.densenet169(pretrained=True)

        self.num_features = self.extravasation_model.classifier.in_features

        for model in [self.extravasation_model,self.bowel_model,self.kidney_model,self.liver_model,self.spleen_model]:
            model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 2)
        )

    def forward(self, x1):

        output1 = self.extravasation_model(x1)
        output2 = self.bowel_model(x1)
        output3 = self.kidney_model(x1)
        output4 = self.liver_model(x1)
        output5 = self.spleen_model(x1)


        output = torch.cat((output1, output2,output3,output4,output5), dim=0)

        return output

def list_extend(y_true,y_pred,labels,preds):
    label = labels.cpu().numpy().tolist()
    pred = preds.cpu().numpy().tolist()
    for index, value in enumerate(label):

        y_true[index].extend(value)
    for index, value in enumerate(pred):
        y_pred[index].append(value)
def evaluate(y_true,y_pred):
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

def main():
    dataset = PNGDataset(r'C:\Users\a6649\Desktop\Honors\Dataset\10_iamge_for_tessing', transform=transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    model = MultiDensennet169().to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # if dataset is imbalanced -> Adam, otherwise -> SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    # Split the data into train and test sets

    train_indices, test_indices = train_test_split(list(range(int(len(dataset)))), test_size=0.4, random_state=123)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=6)

    num_epochs = 40

    for epoch in tqdm(range(num_epochs)):

        running_loss = 0.0
        running_corrects = 0
        model.train()
        try:
#             for img, label in tqdm(train_dataloader):
#                 img = img.to(device)
#                 label = label.squeeze().to(device)
#                 outputs = model(img)
#
#                 loss = criterion(outputs, label)
#
#                 _, preds = torch.max(outputs, 1)
#
#                 optimizer.zero_grad()
#
#                 loss.backward()
#
#                 optimizer.step()
#
#                 running_loss += (loss.item()) * img.size(0)
#                 running_corrects += torch.sum(preds == label.data)
#
# # Here is the overall correctness and accuracy.
#             epoch_loss = running_loss / (len(train_dataset) * 5)
#             epoch_acc = running_corrects.double() / (len(train_dataset) * 5)
#             print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            model.eval()


            y_true = [[],[],[],[],[]]
            y_pred = [[],[],[],[],[]]

            # output_file = save_path + str(epoch) + 'metrics.txt'
            with torch.no_grad():
                for img, label in tqdm(test_dataloader):
                    img = img.to(device)

                    label = label.to(device)

                    outputs = model(img)
                    _, preds = torch.max(outputs, 1)
                    # running_corrects += torch.sum(preds == label.data)

                    print(label,preds)
                    print(label.cpu().numpy().tolist(),preds.cpu().numpy().tolist())

                    list_extend(y_true,y_pred,label,preds)

                print(y_true,y_pred)
                epoch_acc = running_corrects.double() / (len(test_dataset)*5)
                # the reason why it needs to time 5 is that for each image has five label.


                print(f'Acc: {epoch_acc:.4f}')
            # Calculate evaluation metrics
            # Need to evaluate five label individually.



            # with open(output_file, "w") as file:
            #     file.write(f'Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}\n')
            #     file.write(f"F1-Score: {f1:.5f} | Recall: {recall:.2f}\n")
            #     file.write(
            #         f"Specificity: {specificity:.5f} | Sensitivity: {sensitivity:.5f} | Accuracy: {accuracy:.2f}%\n")
        except Exception as e:
            print(e)
            # with open(output_file, "w") as file:
                # file.write(f'exception: {e}')



if __name__ == '__main__':
    main()