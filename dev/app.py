from flask import Flask, render_template, Response, jsonify
import torch
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import cv2
import numpy as np
import time

app = Flask(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def initialize_model(num_classes):
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model = torch.nn.DataParallel(model)
    return model

# Define your classes here
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Create label map
def create_label_map(df):
    labels = df['label'].unique()
    label_map = {label: idx for idx, label in enumerate(labels)}
    return label_map

def predict_image(image, model, transform):
    model.eval()
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

label_map = {label: idx for idx, label in enumerate(classes)}
reverse_label_map = {idx: label for label, idx in label_map.items()}

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

num_classes = len(classes)
model = initialize_model(num_classes)
model.load_state_dict(torch.load('./training/mobilenet_model.pth', map_location=device))
model.to(device)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

text = ""
last_prediction_time = time.time()
prediction_interval = 2

def gen_frames():
    global text, last_prediction_time
    predicted_label = "No prediction yet" 
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:
                # Convert captured frame to PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                predicted_label_idx = predict_image(pil_image, model, transform)
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


@app.route('/',methods=['GET'])
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
    app.run(debug=True,)
