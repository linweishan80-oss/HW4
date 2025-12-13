import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# --- Model Definition ---
# This must be the same architecture as in train.py
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layer(x)
        return x

# --- Helper Functions ---

@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained model weights."""
    model = SimpleCNN()
    # Load the model on CPU, as Streamlit Cloud has limited resources
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode
    return model

def load_labels(labels_path):
    """Loads the class labels from a text file."""
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def predict(image, model, labels, device):
    """Preprocesses the image and makes a prediction."""
    # Define the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = transform(image).unsqueeze(0) # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_idx = torch.max(output, 1)
        
    return labels[predicted_idx.item()]

# --- Streamlit App ---

st.title("CIFAR-10 Image Classification Demo")
st.write("Upload an image, and the model will predict its category.")
st.write("This demo uses a simple CNN trained on the CIFAR-10 dataset.")

# File paths
MODEL_PATH = 'cifar10_model.pth'
LABELS_PATH = 'labels.txt'

# Check for model file
try:
    # Load model and labels
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH)
    model.to(device)
    labels = load_labels(LABELS_PATH)

    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Make prediction
        st.write("Classifying...")
        prediction = predict(image, model, labels, device)
        
        # Display the result
        st.success(f'Prediction: **{prediction}**')

except FileNotFoundError:
    st.error(f"Error: Model file not found at '{MODEL_PATH}'.")
    st.info("Please run the `train.py` script first to train the model and generate the file.")

st.sidebar.header("About")
st.sidebar.info(
    "This is a demo application to showcase deploying a PyTorch deep learning model "
    "with Streamlit. The model is a simple Convolutional Neural Network (CNN) "
    "trained on the CIFAR-10 dataset, which includes 10 classes: airplane, automobile, "
    "bird, cat, deer, dog, frog, horse, ship, and truck."
)
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1.  **Train the model:** If you haven't already, run the training script from your terminal:
    ```bash
    python train.py
    ```
    This will generate the `cifar10_model.pth` file.
2.  **Upload an image:** Use the file uploader in the main panel to select an image.
3.  **Get the prediction:** The model will automatically classify the image and display the result.
""")
