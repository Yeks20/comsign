from flask import Flask, render_template, Response, jsonify 
import torch
from torchvision import transforms, models  
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import time

app = Flask(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def initialize_model(num_classes):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model = torch.nn.DataParallel(model)
    return model

def create_label_map(df):
    labels = df['label'].unique()
    label_map = {label: idx for idx, label in enumerate(labels)}
    return label_map

def predict_image(image, model, transform, label_map):
    model.eval()
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

train_csv = "../data-collection/Training_set.csv"
df = pd.read_csv(train_csv)
label_map = create_label_map(df)
reverse_label_map = {idx: label for label, idx in label_map.items()}


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

num_classes = len(label_map)
model = initialize_model(num_classes)
model.load_state_dict(torch.load("//training/mobilenet_model.h5", map_location=device))
model.to(device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

text = ""
last_prediction_time = time.time()
prediction_interval = 2


def gen_frames():
    global text, last_prediction_time
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:
                # Convert captured frame to PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                predicted_label_idx = predict_image(pil_image, model, transform, label_map)
                predicted_label = reverse_label_map[predicted_label_idx]

                # Update text based on predicted label
                if predicted_label == 'del':
                    text = text[:-1]  # Remove last character
                elif predicted_label == 'space':
                    text += ' '  # Add a space
                elif predicted_label != 'nothing':
                    text += predicted_label  # Add the predicted label

                last_prediction_time = current_time

            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                        cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/text')
def get_text():
    global text
    return text

if __name__ == '__main__':
    app.run(debug=True)
