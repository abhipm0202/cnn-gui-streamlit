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
from datetime import datetime

st.set_page_config(layout="wide", page_title="Colab CNN Trainer")

# --- Config ---
IMAGE_SIZE = (64, 64)
EXTRACT_DIR = "temp_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset Loader ---
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

# --- CNN Builder ---
def build_cnn(n_layers, filters, num_classes):
    layers, in_channels = [], 3
    for _ in range(n_layers):
        layers += [nn.Conv2d(in_channels, filters, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        in_channels = filters
    flat_size = filters * (IMAGE_SIZE[0] // (2**n_layers)) * (IMAGE_SIZE[1] // (2**n_layers))
    layers += [nn.Flatten(), nn.Linear(flat_size, 128), nn.ReLU(), nn.Linear(128, num_classes)]
    return nn.Sequential(*layers)

# --- Training Loop ---
def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    train_loss, val_loss = [], []
    for _ in range(epochs):
        model.train()
        running_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(train_loader))

        model.eval()
        val_running = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                loss = loss_fn(output, y)
                val_running += loss.item()
        val_loss.append(val_running / len(val_loader))
    return train_loss, val_loss

# --- Prediction ---
def predict_image(model, img_file, class_names):
    img = Image.open(img_file).convert("RGB")
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).item()
    return class_names[pred], img

# --- Extract ZIP ---
def extract_zip(zip_file):
    if os.path.exists(EXTRACT_DIR): shutil.rmtree(EXTRACT_DIR)
    with zipfile.ZipFile(zip_file, "r") as z: z.extractall(EXTRACT_DIR)
    return EXTRACT_DIR

# --- Header ---
col1, col2, col3 = st.columns([1.5, 3, 1.5])
with col1:
    st.image("NMIS_logo.png", use_container_width=True)
with col2:
    st.markdown("<h1 style='text-align: center;'>Colab CNN Trainer</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Welcome to CNN GUI developed by D3MColab</h4>", unsafe_allow_html=True)
with col3:
    st.image("Colab_logo.png", use_container_width=True)

# --- Sidebar Config ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    mode = st.radio("Select Mode", ["Train New Model", "Load Trained Model"])

    if mode == "Train New Model":
        uploaded_zip = st.file_uploader("Upload ZIP of labeled folders", type="zip")
        split_ratio = st.slider("Train-Test Split (%)", 10, 90, 80)
        n_layers = st.slider("Conv Layers", 1, 5, 2)
        filters = st.slider("Filters/layer", 8, 128, 32)
        optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "Adagrad"])
        epochs = st.slider("Epochs", 1, 50, 20)
        batch_size = st.slider("Batch Size", 4, 64, 16)
    else:
        model_file = st.file_uploader("Upload trained model (.pt)", type=["pt"])
        label_list = st.text_input("Class Labels (comma-separated)", "Blowhole,Break,Crack,Fray,Free")

# --- Load Model Mode ---
if mode == "Load Trained Model" and model_file is not None:
    try:
        class_names = [cls.strip() for cls in label_list.split(",")]
        model = torch.load(model_file, map_location=DEVICE)
        st.session_state.model = model
        st.session_state.class_names = class_names
        st.success("✅ Model loaded successfully. You can test below.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# --- Train Mode Logic ---
if mode == "Train New Model" and uploaded_zip:
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
        loss_fn = nn.CrossEntropyLoss()

        opt_map = {
            "Adam": (optim.Adam, 0.001),
            "SGD": (optim.SGD, 0.01),
            "RMSprop": (optim.RMSprop, 0.005),
            "Adagrad": (optim.Adagrad, 0.01)
        }
        opt_class, lr = opt_map[optimizer_choice]
        optimizer = opt_class(model.parameters(), lr=lr)

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

        # Prompt to save model
        save_name = st.text_input("💾 Save trained model as (no extension):", f"cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        save_triggered = st.button("Save Trained Model")

        if save_triggered:
            torch.save(model, f"{save_name}.pt")
            st.success(f"Model saved as {save_name}.pt")

            with open(f"{save_name}.pt", "rb") as f:
                st.download_button(
                    label="📥 Download Trained Model",
                    data=f,
                    file_name=f"{save_name}.pt",
                    mime="application/octet-stream"
                )


# --- Prediction UI ---
if "model" in st.session_state and "class_names" in st.session_state:
    st.markdown("---")
    st.subheader("🔍 Try a Prediction")
    test_img = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"])
    if test_img is not None:
        try:
            label, img = predict_image(st.session_state.model, test_img, st.session_state.class_names)
            st.image(img, width=200)
            st.success(f"Predicted Class: {label}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
