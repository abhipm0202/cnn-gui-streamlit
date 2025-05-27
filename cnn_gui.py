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
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, Subset
from collections import Counter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score


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

def predict_image(model, img_file, class_names):
    img = Image.open(img_file).convert("RGB")
    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).item()
    return class_names[pred], img

def extract_zip(zip_file):
    if os.path.exists(EXTRACT_DIR): shutil.rmtree(EXTRACT_DIR)
    with zipfile.ZipFile(zip_file, "r") as z: z.extractall(EXTRACT_DIR)
    return EXTRACT_DIR

# --- Header ---


# --- Header ---
st.markdown(
    '<div style="display: flex; justify-content: space-between; align-items: flex-start; '
    'padding: 0 2rem; margin-bottom: 2rem;">'
    '<img src="https://raw.githubusercontent.com/abhipm0202/cnn-gui-streamlit/main/NMIS_logo.png" '
    'style="height: 70px; margin-top: 5px;" />'
    '<div style="text-align: center; line-height: 1.2;">'
    '<h1 style="margin: 0;">Colab CNN Trainer</h1>'
    '<h4 style="margin: 0;">Welcome to CNN GUI developed by D3MColab</h4>'
    '</div>'
    '<img src="https://raw.githubusercontent.com/abhipm0202/cnn-gui-streamlit/main/Colab_logo.png" '
    'style="height: 80px; margin-top: 5px;" />'
    '</div>',
    unsafe_allow_html=True
)




# --- Sidebar Config ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    mode = st.radio("Select Mode", ["Train New Model", "Load Trained Model"])

    if mode == "Train New Model":
        uploaded_zip = st.file_uploader("Upload ZIP of labeled folders", type="zip")
        split_ratio = st.slider("Train-Test Split (%)", 10, 90, 80)
        use_pretrained = st.checkbox("Use Pretrained Model (ResNet)")
        if use_pretrained:
            resnet_type = st.selectbox("Select ResNet", ["ResNet18", "ResNet50"])
        else:
            n_layers = st.slider("Conv Layers", 1, 20, 3)
            filters = st.slider("Filters/layer", 8, 128, 32)

        optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "Adagrad"])
        epochs = st.slider("Epochs", 1, 50, 20)
        batch_size = st.slider("Batch Size", 4, 64, 16)
    else:
        model_file = st.file_uploader("Upload trained model (.pt)", type=["pt"])
        label_list = st.text_input("Class Labels (comma-separated)", "Blowhole,Break,Crack,Fray,Free")

# --- Load Pretrained Model Mode ---
if mode == "Load Trained Model" and model_file is not None:
    try:
        class_names = [cls.strip() for cls in label_list.split(",")]
        model = torch.load(model_file, map_location=DEVICE, weights_only=False)
        st.session_state.model = model
        st.session_state.class_names = class_names
        st.success("✅ Model loaded successfully. You can test below.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# --- Training Logic ---
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

        if use_pretrained:
            if resnet_type == "ResNet18":
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
        else:
            model = build_cnn(n_layers, filters, len(class_names))

        model = model.to(DEVICE)
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
        st.session_state.trained_model = model
        st.session_state.trained_model_ready = True

        all_preds, all_labels = [], []
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        accuracy = 100.0 * np.trace(cm) / np.sum(cm)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Loss Curve")
            fig, ax = plt.subplots()
            ax.plot(train_loss, label="Train Loss")
            ax.plot(val_loss, label="Validation Loss")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("🧾 Confusion Matrix")
            st.markdown(f"**Accuracy:** {accuracy:.2f}%")
            st.markdown(f"**Precision:** {precision:.2f}")
            st.markdown(f"**Recall:** {recall:.2f}")
            st.markdown(f"**F1-Score:** {f1:.2f}")

            fig2, ax2 = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax2, cmap="Blues")
            ax2.set_xlabel("Predicted Labels")
            ax2.set_ylabel("Actual Labels")
            st.pyplot(fig2)



# --- Save/Download After Training ---
if st.session_state.get("trained_model_ready", False):
    default_name = f"cnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_name = st.text_input("💾 Save trained model as (no extension):", default_name)

    if st.button("Save Trained Model"):
        model_path = f"{save_name}.pt"
        torch.save(st.session_state.trained_model, model_path)
        st.session_state.model_saved = True
        st.session_state.model_path = model_path
        st.success(f"✅ Model saved as {model_path}")

    if st.session_state.get("model_saved", False) and os.path.exists(st.session_state.model_path):
        with open(st.session_state.model_path, "rb") as f:
            st.download_button(
                label="📥 Download Trained Model",
                data=f,
                file_name=st.session_state.model_path,
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
