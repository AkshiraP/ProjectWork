from django.shortcuts import render
from django.http import JsonResponse
import joblib
import os
import gdown
from tensorflow.keras.models import load_model

# Google Drive links (converted to direct download)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1bMajvCTDLrENFINAgp66-zWS2ceUGUzS"
VECTORIZER_URL = "https://drive.google.com/uc?export=download&id=1QUcO9y4YL1kWr_vtOPabEIrckD5zN5RA"

# Base directory and paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, 'myapp', 'models')
model_path = os.path.join(models_dir, 'ToxicityModel.h5')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

# Download files if missing
if not os.path.exists(model_path):
    print("Downloading Keras model from Google Drive...")
    gdown.download(MODEL_URL, model_path, quiet=False)

if not os.path.exists(vectorizer_path):
    print("Downloading vectorizer from Google Drive...")
    gdown.download(VECTORIZER_URL, vectorizer_path, quiet=False)

# Load the Keras model and vectorizer once at startup
model = load_model(model_path)
vectorizer = joblib.load(vectorizer_path)

# Define class labels for multi-label classification
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
threshold = 0.5  # Adjust if needed (e.g., 0.3 to be more sensitive)

# Home page view
def home(request):
    return render(request, 'myapp/home.html')

# Comment classification API view
def classify_comment(request):
    if request.method == 'POST':
        comment = request.POST.get('comment')

        if comment:
            # Preprocess comment
            comment = comment.lower().strip()

            # Vectorize and predict
            vectorized_comment = vectorizer([comment])
            prediction = model.predict(vectorized_comment)  # returns shape (1, 6)

            # Convert prediction to list
            prediction = prediction[0].tolist()

            # Calculate toxicity score — max predicted probability
            toxicity_score = max(prediction)

            # Determine labels above threshold
            toxic_labels = [labels[i] for i, prob in enumerate(prediction) if prob > threshold]

            # Result
            if toxic_labels:
                result = f"Toxic ({', '.join(toxic_labels)})"
            else:
                result = "Non-Toxic"

            return JsonResponse({
                'result': result,
                'score': float(toxicity_score)
            })

    return JsonResponse({'result': 'Invalid input'})
