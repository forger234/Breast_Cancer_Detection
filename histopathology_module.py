import os # Import the built-in os module for interacting with the operating system
import numpy as np # Import NumPy with the standard 'np' alias for numerical computations and array manipulation
import joblib # Import joblib for loading the serialized machine learning model from disk
import streamlit as st # Import the streamlit library to build the interactive web application interface
import tf_keras as keras # Import tf_keras to access deep learning models and functions
from tf_keras.preprocessing import image # Import the image module from tf_keras for image loading and preprocessing tools
from tf_keras.applications.efficientnet import preprocess_input # Import the EfficientNet-specific preprocessing function

CLASSES = ["Benign", "Malignant", "Normal"] # Define the target classification labels representing the three possible outcomes

@st.cache_resource # Use Streamlit's cache_resource decorator to ensure models are loaded only once and cached across reruns
def load_histo_models(): # Define the function responsible for loading the histopathology models into memory
    # The paths are inside 'bc' folder # Comment indicating that model files are located in the 'bc' directory
    svm_path = os.path.join("bc", "svm_bach.joblib") # Construct an OS-independent path to the saved SVM model file
    svm = joblib.load(svm_path) # Load the trained Support Vector Machine (SVM) model using joblib
    feature_extractor = keras.applications.EfficientNetB0( # Initialize the EfficientNetB0 pre-trained model to extract image features
        weights="imagenet", # Load the model with pre-trained weights from the ImageNet dataset
        include_top=False, # Exclude the final fully-connected classification layer of the EfficientNet model
        pooling="avg", # Apply global average pooling to the final tensor, reducing spatial dimensions to a 1D vector
        input_shape=(224, 224, 3) # Specify the required input shape for EfficientNetB0 (224x224 pixels, 3 color channels)
    ) # Close the model initialization arguments
    return svm, feature_extractor # Return the loaded SVM and feature extractor for use in prediction

def predict_histopathology(pil_img): # Define the main prediction function taking a PIL image as input
    svm, feature_extractor = load_histo_models() # Call the loading function to retrieve the cached models
    
    # Needs (224, 224) # Comment reminding that the input image must be resized to 224x224 pixels
    if pil_img.mode != "RGB": # Check if the provided image is not in standard RGB color mode
        pil_img = pil_img.convert("RGB") # Convert the image to RGB to ensure compatibility with the models (removes alpha channels etc.)
    
    img = pil_img.resize((224, 224)) # Resize the image to 224x224 pixels as expected by EfficientNetB0
    img_array = image.img_to_array(img) # Convert the PIL Image object into a multidimensional NumPy array
    img_array = preprocess_input(img_array) # Apply the necessary pixel value scaling specific to EfficientNet models
    
    # Predict features using the callable directly instead of .predict() for much faster single-image inference # Comment explaining performance optimization
    features = feature_extractor(np.expand_dims(img_array, 0), training=False).numpy() # Expand array dimensions to add a batch axis, extract features, and convert to numpy array
    probs = svm.predict_proba(features)[0] # Pass the extracted features to the SVM to predict probabilities, selecting the first result
    
    best = np.argmax(probs) # Find the index corresponding to the highest predicted probability
    confidence = probs[best] * 100.0 # Calculate the confidence score as a percentage by multiplying the top probability by 100
    pred_name = CLASSES[best].lower() # Retrieve the correspondingly named class label from CLASSES list and convert to lowercase (e.g., "benign", "malignant", "normal")
    
    return pred_name, confidence, probs # Return the final predicted name, the calculated confidence, and the raw probability array for all classes
