import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from flask import Flask, request, render_template, url_for
from PIL import Image
import os
from efficientnet_mini import EfficientNetMini

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load EfficientNetMini model
try:
    efficientnet_model = EfficientNetMini()
    try:
        efficientnet_model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu'), weights_only=True))
    except TypeError:
        # Fallback for older versions
        efficientnet_model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    efficientnet_model.eval()
    print("EfficientNetMini model loaded successfully.")
except Exception as e:
    print("Error loading EfficientNetMini model:", e)

# Load class names for EfficientNetMini model
try:
    with open("class_names.txt", "r") as file:
        class_names = [line.strip() for line in file]
    print(f"Class names loaded ({len(class_names)} classes): {class_names}")
except Exception as e:
    print("Error loading class names:", e)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Standard size for EfficientNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to classify the input image
def classify_image(image):
    try:
        print("Starting image classification...")

        # Print the original image size
        print("Original image size:", image.size)

        # Transform the image
        image = transform(image).unsqueeze(0)  # Add batch dimension
        print("Transformed image shape:", image.shape)

        # Ensure the model is in evaluation mode
        efficientnet_model.eval()

        # Perform inference
        with torch.no_grad():
            output = efficientnet_model(image)
            print("Model output shape:", output.shape)

            # If the output is a 1D tensor, handle it directly
            if len(output.shape) == 1:
                probabilities = F.softmax(output, dim=0)
                predicted_index = torch.argmax(probabilities).item()
            elif len(output.shape) == 2:
                probabilities = F.softmax(output, dim=1)
                predicted_index = torch.argmax(probabilities, dim=1).item()
            else:
                print("Unexpected output shape:", output.shape)
                return "Unexpected output shape"

        # Print the class probabilities for debugging
        print("Class probabilities:", probabilities.tolist())
        print("Predicted class index:", predicted_index)

        # Check if the predicted index is within bounds
        if 0 <= predicted_index < len(class_names):
            return class_names[predicted_index]
        else:
            print("Predicted index is out of bounds.")
            return "Invalid prediction"

    except Exception as e:
        print("Error in classify_image:", e)
        return "Classification error"

# Route for the home page
@app.route('/', methods=['GET'])
def index():
    return render_template('disease.html', message="Upload an image to predict plant disease.")

# Route for handling file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return render_template('disease.html', message="No file part.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('disease.html', message="No selected file.")

        # Save the uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Open and preprocess the image
        pil_image = Image.open(file_path).convert("RGB")
        prediction = classify_image(pil_image)

        return render_template('disease.html', message=f"Prediction: {prediction}", image_path=url_for('static', filename='uploads/' + file.filename))
    except Exception as e:
        print("Error in predict route:", e)
        return render_template('disease.html', message="An error occurred during prediction.")

if __name__ == '__main__':
    app.run(debug=True)
