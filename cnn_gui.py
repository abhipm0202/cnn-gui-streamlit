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

# --- Config ---
IMAGE_SIZE = (64, 64)
EXTRACT_DIR = "temp_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset Loader ---
class ImageFolderDataset(Dataset):
    def __init__(self, base_dir, class_to_idx):
        self.paths = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        for label, idx in class_to_idx.items():
            folder = os.path.join(base_dir, label)
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                if fname.lower().endswith(("png", "jpg", "jpeg")):
                    self.paths.append(path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]

# --- CNN Builder ---
def build_cnn(n_layers, filters, num_classes):
    layers = []
    in_channels = 3
    for _ in range(n_layers):
        layers.append(nn.Conv2d(in_channels, filters, 3, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        in_channels = filters
    flat_size = filters * (IMAGE_SIZE[0] // (2**n_layers)) * (IMAGE_SIZE[1] // (2**n_layers))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(flat_size, 128))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(128, num_classes))
    return nn.Sequential(*layers)

# --- Training ---
def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(train_loader))

        model.eval()
        val_running = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = loss_fn(out, y)
                val_running += loss.item()
        val_loss.append(val_running / len(val_loader))
    return train_loss, val_loss

# --- Predict One Image ---
def predict_image(model, img_path, class_names):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor).argmax(dim=1).item()
    return class_names[pred], img

# --- Zip Extract ---
def extract_zip(zip_file):
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    return EXTRACT_DIR

# --- Streamlit App ---
st.title("CNN Image Classifier (PyTorch + Streamlit)")

uploaded_zip = st.file_uploader("Upload ZIP of labeled image folders", type="zip")
if uploaded_zip:
    base_dir = extract_zip(uploaded_zip)
    class_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    dataset = ImageFolderDataset(base_dir, class_to_idx)

    # Show sample counts
    label_counts = Counter(dataset.labels)
    st.write("📊 Image count per class:", {class_names[k]: v for k, v in label_counts.items()})

    split_ratio = st.slider("Train-Test Split (%)", 10, 90, 80)
    n_layers = st.slider("Conv Layers", 1, 5, 2)
    filters = st.slider("Filters per Layer", 8, 128, 32)
    optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD"])
    epochs = st.slider("Epochs", 1, 50, 10)
    batch_size = st.slider("Batch Size", 4, 64, 16)

    if st.button("Train CNN"):
        # Stratified split
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=(100 - split_ratio) / 100,
            stratify=dataset.labels,
            random_state=42
        )
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

        model = build_cnn(n_layers, filters, len(class_names)).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters()) if optimizer_choice == "Adam" else optim.SGD(model.parameters(), lr=0.01)

        train_loss, val_loss = train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs)
        st.session_state.model = model
        st.session_state.class_names = class_names

        # Loss Curve
        st.subheader("📉 Loss Over Epochs")
        fig, ax = plt.subplots()
        ax.plot(train_loss, label="Train Loss")
        ax.plot(val_loss, label="Validation Loss")
        ax.legend()
        st.pyplot(fig)

        # Confusion Matrix
        all_preds, all_labels = [], []
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                out = model(x)
                preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax2)
        st.subheader("Confusion Matrix")
        st.pyplot(fig2)

# --- Predict ---
if "model" in st.session_state:
    st.header("Test with a New Image")
    test_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if test_img:
        label, img = predict_image(st.session_state.model, test_img, st.session_state.class_names)
        st.image(img, width=200)
        st.success(f"Predicted Class: {label}")
