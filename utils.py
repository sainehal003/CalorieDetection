import numpy as np
import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

# ---------------------------
# Load ImageNet MobileNetV2 Model
# ---------------------------
model = MobileNetV2(weights='imagenet', include_top=True)

# ---------------------------
# Preprocess Image
# ---------------------------
def preprocess_image(img):
    """Accepts a PIL Image, returns preprocessed tensor."""
    if isinstance(img, Image.Image):
        img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# ---------------------------
# Predict Food
# ---------------------------
def predict_food(img, top_k=1):
    """
    Predict food name & probability from an image using MobileNetV2.
    Returns the top prediction from the ImageNet model.
    """
    x = preprocess_image(img)
    preds = model.predict(x)
    # decode_predictions returns a list of lists of tuples `(class_id, description, probability)`
    decoded = decode_predictions(preds, top=top_k)[0]

    # Get the top prediction
    top_prediction = decoded[0]
    food_name = top_prediction[1].replace("_", " ")
    confidence = top_prediction[2] * 100
    
    return food_name, confidence

# ---------------------------
# Get Calories via Open Food Facts API
# ---------------------------
def get_calories(food_name):
    """
    Return calorie info from the Open Food Facts API.
    Searches for the food name and returns calories per 100g.
    """
    # Format the food name for the API URL
    search_term = food_name.lower().strip()
    
    # Construct the API request URL
    url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={search_term}&search_simple=1&action=process&json=1&page_size=1"
    
    headers = {
        'User-Agent': 'FoodCalorieEstimatorApp/1.0'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()

        # Check if any products were found and if the first product has calorie info
        if data.get('products') and 'nutriments' in data['products'][0]:
            nutriments = data['products'][0]['nutriments']
            # The API provides energy in kcal per 100g
            if 'energy-kcal_100g' in nutriments:
                return int(nutriments['energy-kcal_100g'])
            elif 'energy_100g' in nutriments: # Fallback to energy in kJ and convert
                return int(nutriments['energy_100g'] * 0.239006)

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
    except (ValueError, KeyError, IndexError):
        # Handles JSON parsing errors or missing keys
        print("Failed to parse calorie data from API response.")

    return None
