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
from torch.utils.data import DataLoader, Dataset, Subset
from collections import Counter

IMAGE_SIZE = (64, 64)
EXTRACT_DIR = "temp_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ImageFolderDataset(Dataset):
    def __init__(self, base_dir, class_to_idx):
        self.paths, self.labels = [], []
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        for label, idx in class_to_idx.items():
            folder = os.path.join(base_dir, label)
            for fname in os.listdir(folder):
                if fname.lower().endswith(("png", "jpg", "jpeg")):
                    self.paths.append(os.path.join(folder, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

def build_cnn(n_layers, filters, num_classes):
    layers, in_channels = [], 3
    for _ in range(n_layers):
        layers += [nn.Conv2d(in_channels, filters, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        in_channels = filters
    layers.append(nn.Flatten())
    flat_size = filters * (IMAGE_SIZE[0] // (2**n_layers)) * (IMAGE_SIZE[1] // (2**n_layers))
    layers.append(nn.Linear(flat_size, 128))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(128, num_classes))
    return nn.Sequential(*layers)

def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        model.train()
        t_loss = sum(loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).item() 
                     for x, y in train_loader)
        train_loss.append(t_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            v_loss = sum(loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).item()
                         for x, y in val_loader)
        val_loss.append(v_loss / len(val_loader))
    return train_loss, val_loss

def predict_image(model, img_path, class_names):
    img = Image.open(img_path).convert("RGB")
    tensor = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).item()
    return class_names[pred], img

def extract_zip(zip_file):
    if os.path.exists(EXTRACT_DIR): shutil.rmtree(EXTRACT_DIR)
    with zipfile.ZipFile(zip_file, "r") as z: z.extractall(EXTRACT_DIR)
    return EXTRACT_DIR

# --- Header UI ---
logo_col1, title_col, logo_col2 = st.columns([1, 4, 1])
with logo_col1:
    st.image("NMIS_logo.png", width=100)
with title_col:
    st.title("Colab CNN Trainer")
    st.write("Welcome to CNN GUI developed by D3MColab")
with logo_col2:
    st.image("Colab_logo.png", width=100)

# --- Columns layout ---
left_col, right_col = st.columns([1, 2])

with left_col:
    uploaded_zip = st.file_uploader("Upload ZIP of labeled image folders", type="zip")
    if uploaded_zip:
        base_dir = extract_zip(uploaded_zip)
        class_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        dataset = ImageFolderDataset(base_dir, class_to_idx)
        st.write("📊 Class Sample Counts:", {class_names[i]: c for i, c in Counter(dataset.labels).items()})

        split_ratio = st.slider("Train-Test Split (%)", 10, 90, 80)
        n_layers = st.slider("Conv Layers", 1, 5, 2)
        filters = st.slider("Filters per Conv Layer", 8, 128, 32)

        optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "Adagrad"])
        loss_choice = st.selectbox("Loss Function", ["CrossEntropyLoss", "NLLLoss", "MSELoss"])
        epochs = st.slider("Epochs", 1, 50, 10)
        batch_size = st.slider("Batch Size", 4, 64, 16)

with right_col:
    if uploaded_zip and st.button("🚀 Train CNN"):
        train_idx, val_idx = train_test_split(
            list(range(len(dataset))),
            test_size=(100 - split_ratio) / 100,
            stratify=dataset.labels,
            random_state=42
        )
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

        model = build_cnn(n_layers, filters, len(class_names)).to(DEVICE)

        loss_fn = getattr(nn, loss_choice)()
        opt_map = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "RMSprop": optim.RMSprop,
            "Adagrad": optim.Adagrad
        }
        optimizer = opt_map[optimizer_choice](model.parameters(), lr=0.001)

        train_loss, val_loss = train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs)
        st.session_state.model = model
        st.session_state.class_names = class_names

        st.subheader("📉 Loss Curve")
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="Train Loss")
        ax.plot(val_loss, label="Validation Loss")
        ax.legend()
        st.pyplot(fig)

        # Confusion matrix
        all_preds, all_labels = [], []
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        cm = confusion_matrix(all_labels, all_preds)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax2)
        st.subheader("Confusion Matrix")
        st.pyplot(fig2)

    if "model" in st.session_state:
        st.subheader("🔍 Test Trained Model")
        test_img = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="predict")
        if test_img:
            label, img = predict_image(st.session_state.model, test_img, st.session_state.class_names)
            st.image(img, width=200)
            st.success(f"Predicted Class: {label}")
