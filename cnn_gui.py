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

st.set_page_config(layout="wide", page_title="Colab CNN Trainer")

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

    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

def build_cnn(n_layers, filters, num_classes):
    layers, in_channels = [], 3
    for _ in range(n_layers):
        layers += [nn.Conv2d(in_channels, filters, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        in_channels = filters
    flat_size = filters * (IMAGE_SIZE[0] // (2**n_layers)) * (IMAGE_SIZE[1] // (2**n_layers))
    layers += [nn.Flatten(), nn.Linear(flat_size, 128), nn.ReLU(), nn.Linear(128, num_classes)]
    return nn.Sequential(*layers)

def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    train_loss, val_loss = [], []
    for _ in range(epochs):
        model.train()
        total = sum(loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).item() for x, y in train_loader)
        train_loss.append(total / len(train_loader))
        model.eval()
        with torch.no_grad():
            vtotal = sum(loss_fn(model(x.to(DEVICE)), y.to(DEVICE)).item() for x, y in val_loader)
        val_loss.append(vtotal / len(val_loader))
    return train_loss, val_loss

def predict_image(model, img_file, class_names):
    img = Image.open(img_file).convert("RGB")
    tensor = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).item()
    return class_names[pred], img

def extract_zip(zip_file):
    if os.path.exists(EXTRACT_DIR): shutil.rmtree(EXTRACT_DIR)
    with zipfile.ZipFile(zip_file, "r") as z: z.extractall(EXTRACT_DIR)
    return EXTRACT_DIR

# --- Header ---
col1, col2, col3 = st.columns([1.5, 3, 1.5])
with col1:
    st.image("NMIS_logo.png", use_column_width=True)
with col2:
    st.markdown("<h1 style='text-align: center;'>Colab CNN Trainer</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Welcome to CNN GUI developed by D3MColab</h4>", unsafe_allow_html=True)
with col3:
    st.image("Colab_logo.png", use_column_width=True)

# --- Sidebar for inputs ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    uploaded_zip = st.file_uploader("Upload ZIP of labeled folders", type="zip")
    split_ratio = st.slider("Train-Test Split (%)", 10, 90, 80)
    n_layers = st.slider("Conv Layers", 1, 5, 2)
    filters = st.slider("Filters/layer", 8, 128, 32)
    optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "Adagrad"])
    loss_choice = st.selectbox("Loss Function", ["CrossEntropyLoss", "NLLLoss", "MSELoss"])
    epochs = st.slider("Epochs", 1, 50, 10)
    batch_size = st.slider("Batch Size", 4, 64, 16)

# --- Training Panel (Main Body) ---
if uploaded_zip:
    base_dir = extract_zip(uploaded_zip)
    class_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    dataset = ImageFolderDataset(base_dir, class_to_idx)
    st.markdown("### 📊 Class Sample Counts:")
    st.json({class_names[i]: c for i, c in Counter(dataset.labels).items()})

    if st.button("🚀 Train CNN"):
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

        st.info("Training in progress...")
        train_loss, val_loss = train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs)
        st.session_state.model = model
        st.session_state.class_names = class_names

        st.subheader("📉 Loss Curve")
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="Train Loss")
        ax.plot(val_loss, label="Validation Loss")
        ax.legend()
        st.pyplot(fig)

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
        st.subheader("🧾 Confusion Matrix")
        st.pyplot(fig2)

# --- Prediction Panel ---
if "model" in st.session_state:
    st.markdown("---")
    st.subheader("🔍 Try a Prediction")
    test_img = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"], key="predict")
    if test_img:
        label, img = predict_image(st.session_state.model, test_img, st.session_state.class_names)
        st.image(img, width=200)
        st.success(f"Predicted Class: {label}")
