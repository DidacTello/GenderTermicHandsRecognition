import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class ThermalHandsDataset(torch.utils.data.Dataset):
    def __init__(self, base_dirs, annotation_file, transform=None, images_per_person=10):
        self.annotations = pd.read_csv(annotation_file, header=None, sep=" ", names=['id', 'gender'])
        self.transform = transform
        self.images_per_person = images_per_person
        self.image_paths = []
        self.labels = []

        for base_dir in base_dirs:
            for _, row in self.annotations.iterrows():
                person_id, gender = row['id'], row['gender']
                person_folder = f"s{person_id}"
                full_path = os.path.join(base_dir, person_folder)
                if os.path.isdir(full_path):
                    person_images = sorted(os.listdir(full_path))[:self.images_per_person]

                    for img in person_images:
                        if img != "Thumbs.db":
                            self.image_paths.append(os.path.join(full_path, img))
                            self.labels.append(0 if gender == 'Home' else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label




transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


base_dirs = ['TermicPalm/Termica', 'TermicDors/Termica']
dataset = ThermalHandsDataset(base_dirs=base_dirs, annotation_file='sexes.txt', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = models.vgg16(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_ftrs, 2)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()


def train_and_validate(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))


        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
        val_losses.append(running_loss / len(test_loader))

        print(f"Epoch {epoch+1}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")

    return train_losses, val_losses


def plot_and_save_losses(train_losses, val_losses, filename="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Evolution of Loss During Training and Validation')
    plt.legend()
    plt.savefig(filename)
    plt.show()


train_losses, val_losses = train_and_validate(model, train_loader, test_loader, criterion, optimizer, num_epochs=4)


plot_and_save_losses(train_losses, val_losses, filename="training_validation_loss.png")



def evaluate_and_visualize(model, loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred, y_true = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())


    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks([0.5, 1.5], ['Home', 'Dona'])
    plt.yticks([0.5, 1.5], ['Home', 'Dona'])
    plt.title('Confusion Matrix')
    plt.show()


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')

    return y_pred, y_true


def save_confusion_matrix(loader, model, filename="confusion_matrix.png"):
    y_pred, y_true = evaluate_and_visualize(model, loader)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks([0.5, 1.5], ['Home', 'Dona'])
    plt.yticks([0.5, 1.5], ['Home', 'Dona'])
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()


save_confusion_matrix(test_loader, model, filename="confusion_matrix.png")


def show_sample_predictions(loader, model, num_images=5, filename="sample_predictions.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"True: {labels[i].item()}, Pred: {preds[i].item()}")
        plt.axis('off')
    
    plt.savefig(filename)
    plt.show()

show_sample_predictions(test_loader, model, num_images=5, filename="sample_predictions.png")