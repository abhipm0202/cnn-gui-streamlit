import streamlit as st
import zipfile, os, shutil
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# ---------- CONFIG ----------
IMAGE_SIZE = (64, 64)
EXTRACT_DIR = "temp_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- DATASET CLASS ----------
class ImageFolderDataset(Dataset):
    def __init__(self, base_dir, class_to_idx):
        self.paths = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        for label, idx in class_to_idx.items():
            full_path = os.path.join(base_dir, label)
            for fname in os.listdir(full_path):
                self.paths.append(os.path.join(full_path, fname))
                self.labels.append(idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

# ---------- CNN DEFINITION ----------
def build_cnn(n_layers, filters, num_classes):
    layers = []
    in_channels = 3
    for _ in range(n_layers):
        layers.append(nn.Conv2d(in_channels, filters, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        in_channels = filters
    layers.append(nn.Flatten())
    layers.append(nn.Linear(filters * (IMAGE_SIZE[0] // (2**n_layers)) * (IMAGE_SIZE[1] // (2**n_layers)), 128))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(128, num_classes))
    return nn.Sequential(*layers)

# ---------- ZIP EXTRACTION ----------
def extract_zip(zip_file):
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(EXTRACT_DIR)
    return EXTRACT_DIR

# ---------- TRAIN FUNCTION ----------
def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss.append(total_loss / len(train_loader))

        model.eval()
        total_vloss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                total_vloss += loss.item()
        val_loss.append(total_vloss / len(val_loader))
    return train_loss, val_loss

# ---------- PREDICTION ----------
def predict_image(model, img_path, class_names):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return class_names[pred], img

# ---------- STREAMLIT UI ----------
st.title("CNN Trainer and Inference App (PyTorch)")

uploaded_zip = st.file_uploader("Upload ZIP file of labeled image folders", type="zip")
if uploaded_zip:
    data_dir = extract_zip(uploaded_zip)
    class_names = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    dataset = ImageFolderDataset(data_dir, class_to_idx)

    split_ratio = st.slider("Train/Test Split (%)", 10, 90, 80)
    n_layers = st.slider("Number of Conv Layers", 1, 5, 2)
    filters = st.slider("Filters per Conv Layer", 8, 128, 32)
    loss_choice = st.selectbox("Loss Function", ["CrossEntropyLoss"])
    opt_choice = st.selectbox("Optimizer", ["Adam", "SGD"])
    epochs = st.slider("Number of Epochs", 1, 50, 5)
    batch_size = st.slider("Batch Size", 4, 64, 16)

    if st.button("Train CNN"):
        indices = list(range(len(dataset)))
        train_size = int(len(indices) * split_ratio / 100)
        train_indices, val_indices = indices[:train_size], indices[train_size:]
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_indices))

        model = build_cnn(n_layers, filters, len(class_names)).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters()) if opt_choice == "Adam" else optim.SGD(model.parameters(), lr=0.01)

        st.write("Training model...")
        train_loss, val_loss = train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs)

        st.session_state.model = model
        st.session_state.class_names = class_names

        st.subheader("Loss Curve")
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="Train Loss")
        ax.plot(val_loss, label="Val Loss")
        ax.legend()
        st.pyplot(fig)

        # Confusion Matrix
        all_preds, all_labels = [], []
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax2)
        st.subheader("Confusion Matrix")
        st.pyplot(fig2)

# Prediction
if "model" in st.session_state:
    st.header("Test the trained CNN")
    test_img = st.file_uploader("Upload an image for prediction", type=["jpg", "png"])
    if test_img:
        pred_label, img = predict_image(st.session_state.model, test_img, st.session_state.class_names)
        st.image(img, width=200)
        st.success(f"Predicted Class: {pred_label}")
