# comsign
### Establishing a sign language identification system through empowering people who communicate using sign language, eliminating communication gaps, increasing accessibility to information, and encouraging equitable participation

### Installation
### Clone the repository
### Running the Application
### Navigate to the cloned repository's directory.
### Run the Flask application:

### How It Works
###  The application initializes a Flask server.
### The video stream is captured from the default webcam.
### Every 2 seconds, a frame is captured and passed through the model to predict a label.
### The predicted label is displayed on the video frame.
### The video stream and the current text prediction are available through the web interface.

## Routes
### `/`: The main index route that renders the web interface.
### `/video_feed`: Streams the video feed with predictions.
### `/text`: Returns the current text prediction.

## Model Details
### The model is a MobileNetV2 architecture, modified to predict a custom number of classes based on the training data.
### The model weights are loaded from a `.h5` file.
### A label map is created from a CSV file containing training data labels.

## Notes
### Ensure the `mobilenet_model.h5` file is located in the `training` directory.
### The CSV file for creating the label map should be in the `data-collection` directory.
### The application uses the CPU by default, but if a CUDA-compatible GPU is available, it will use it for faster predictions.





