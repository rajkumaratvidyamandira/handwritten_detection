from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np


label_map = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
    10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's',
    19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: 'A', 27: 'B',
    28: 'C', 29: 'D', 30: 'E', 31: 'F', 32: 'G', 33: 'H', 34: 'I', 35: 'J', 36: 'K',
    37: 'L', 38: 'M', 39: 'N', 40: 'O', 41: 'P', 42: 'Q', 43: 'R', 44: 'S', 45: 'T',
    46: 'U'
}

class CNN(nn.Module):
    def __init__(self, num_classes=47):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
try:
    model = CNN(num_classes=47)
    model.load_state_dict(torch.load('model/CNN/emnist_cnn_final.pth', map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    print("Error: Model file 'model/best_emnist_cnn.pth' not found.")
    raise
except RuntimeError as e:
    print("Error: Model architecture mismatch or corrupted weights.")
    raise

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: TF.rotate(x, 90)),
])

def app():
    st.title("Hand Written Alphabet detect")
    st.write("---")
    col1,col2 = st.columns(2)

    with col1:
        canvas_result = st_canvas(
            stroke_width=25,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )


    with col2:
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
            pil_image = Image.fromarray(img)
            st.image(pil_image.resize((28,28)),width=100, caption="28x28")
            input_tensor = transform(pil_image).unsqueeze(0)

            flag = st.button("Predict")
            
            if flag:
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    pred = torch.argmax(probs).item()
                    confidence = probs[pred].item()

                st.title(f"predected: {label_map[pred]}")

