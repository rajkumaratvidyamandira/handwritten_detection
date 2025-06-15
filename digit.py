import streamlit as st
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class CNN(nn.Module):
    def __init__(self):
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
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    

try:
    model = CNN()
    model.load_state_dict(torch.load('model/CNN/best_mnist_cnn.pth', map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    print("Error: Model file 'model/best_mnist_cnn.pth' not found.")
    raise
except RuntimeError as e:   
    print("Error: Model architecture mismatch or corrupted weights. Check model definition and weights file.")
    raise

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def app():
    st.title("Hand Written Digit detect")
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

                st.title(f"predected: {pred}")

