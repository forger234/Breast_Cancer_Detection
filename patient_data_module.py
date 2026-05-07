import numpy as np # Import the NumPy library for handling arrays and numerical operations
from histopathology.histopathology_module import predict_histopathology # Import the predict_histopathology function from the corresponding local module

def predict_patient_data(us_pil_img, histo_pil_img, us_model, get_gradcam_overlay, class_names): # Define function to perform predictions combining ultrasound and histopathology
    """ # Open the docstring block describing the function
    Processes both Ultrasound and Histopathology images to produce a combined prediction. # High-level description of what the function does
    
    Args: # Start listing the expected arguments for the function
        us_pil_img: PIL Image of the ultrasound. # Argument doc for the ultrasound PIL image
        histo_pil_img: PIL Image of the histopathology. # Argument doc for the histopathology PIL image
        us_model: The loaded ResNet18 model for ultrasound. # Argument doc for the pre-loaded ultrasound PyTorch model
        get_gradcam_overlay: The function to get Grad-CAM and ultrasound predictions. # Argument doc for the utility function producing Grad-CAM overlays
        class_names: List of class names, e.g., ["normal", "benign", "malignant"]. # Argument doc detailing the list of expected string class labels
        
    Returns: # Describe the return values
        dict: A dictionary containing individual and combined results. # Explain that it returns a nested dictionary with multiple prediction scopes
    """ # Close the function docstring
    
    # 1. Ultrasound Prediction # Comment head indicating the start of the ultrasound prediction logic block
    us_overlay_np, us_pred_name, us_probs = get_gradcam_overlay(us_pil_img, us_model) # Execute ultrasound prediction and overlay generation and store returned tuple elements
    
    # 2. Histopathology Prediction # Comment head indicating the start of the histopathology prediction logic block
    # Note: predict_histopathology returns (pred_name, confidence, probs) # Developer note explaining the return structure of the histopathology pipeline
    # where probs are in order: ["Benign", "Malignant", "Normal"] (from CLASSES in that module) # Developer note documenting the distinct probability ordering mismatch
    histo_pred_name, histo_confidence, histo_probs_raw = predict_histopathology(histo_pil_img) # Run the histopathology model on the image, capturing prediction data
    
    # Reorder Histopathology probs to match Ultrasound (CLASS_NAMES = ["normal", "benign", "malignant"]) # Comment explaining the need to align array indices representing classes
    # histo_probs_raw: [Benign, Malignant, Normal] -> Indices: [0, 1, 2] # Comment illustrating the original histopathology index mapping
    # Target: [Normal, Benign, Malignant] # Comment illustrating the target ultrasound index mapping
    histo_probs = np.zeros(3) # Create a new zero-filled NumPy array of size 3 to hold the reordered specific probabilities
    histo_probs[0] = histo_probs_raw[2] # Assign the 'Normal' probability (index 2 in raw) to index 0 in the new array
    histo_probs[1] = histo_probs_raw[0] # Assign the 'Benign' probability (index 0 in raw) to index 1 in the new array
    histo_probs[2] = histo_probs_raw[1] # Assign the 'Malignant' probability (index 1 in raw) to index 2 in the new array
    
    # 3. Combined Prediction (Average Probabilities) # Comment indicating the start of the aggregation logic block
    # Giving equal weight (50/50) to both modalities # Comment detailing the basic ensemble logic utilizing unweighted averages
    combined_probs = (us_probs + histo_probs) / 2.0 # Calculate the element-wise average of both sets of matched probabilities arrays
    
    combined_pred_idx = int(np.argmax(combined_probs)) # Find the integer index representing the maximum value within the averaged probabilities
    combined_pred_name = class_names[combined_pred_idx] # Lookup the textual name associated with the winning index using the shared class mapping
    combined_confidence = combined_probs[combined_pred_idx] * 100.0 # Convert the winning probability decimal into a percentage representation
    
    # Round individual confidence for Ultrasound # Comment header specifying formatting modifications for the ultrasound confidence score
    us_pred_idx = class_names.index(us_pred_name) # Derive the index associated with the selected ultrasound prediction string
    us_confidence = us_probs[us_pred_idx] * 100.0 # Convert the raw ultrasound decimal probability into a fractional percentage format
    
    return { # Initiate the return statement mapping data into the structured output dictionary
        "combined": { # Create a section in the dictionary specifically for aggregated predictions
            "prediction": combined_pred_name, # Map the calculated combined label string
            "confidence": combined_confidence, # Map the overall combined prediction confidence percentage
            "probabilities": combined_probs # Map the generated mathematical array of the combined likelihoods
        }, # Close the combined section block
        "ultrasound": { # Create a section profiling specific standalone ultrasound outcomes
            "prediction": us_pred_name, # Map the standalone ultrasound category string
            "confidence": us_confidence, # Map the standalone ultrasound confidence percentage measure
            "probabilities": us_probs, # Map the original raw probabilistic outputs mapped for ultrasound analysis
            "overlay": us_overlay_np # Map the generated visual array illustrating Grad-CAM highlighted areas
        }, # Close the standalone ultrasound section
        "histopathology": { # Create a final section profiling specific standalone histopathology metrics
            "prediction": histo_pred_name, # Map the direct textual label computed over histopathology
            "confidence": histo_confidence, # Map the explicitly calculated histopathology outcome confidence 
            "probabilities": histo_probs # Map the manually reordered array of probabilities ensuring schema consistency globally
        } # Close the standalone histopathology section
    } # Close the overall returned structure representing explicit aggregated evaluations
