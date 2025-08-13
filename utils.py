import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image

# ---------------------------
# Load ImageNet MobileNetV2 Model
# ---------------------------
model = MobileNetV2(weights='imagenet', include_top=True)

# ---------------------------
# Load Calorie Data
# ---------------------------
labels_path = os.path.join("data", "calorie_data.csv")
calorie_df = pd.read_csv(labels_path)
calorie_df = calorie_df.drop_duplicates(subset=['food'])  # remove duplicate rows
food_list = calorie_df["food"].str.lower().tolist()

# ---------------------------
# Preprocess Image
# ---------------------------
def preprocess_image(img):
    """Accepts file path or PIL Image, returns preprocessed tensor."""
    if isinstance(img, str):  # file path
        img = image.load_img(img, target_size=(224, 224))
    elif isinstance(img, Image.Image):  # PIL Image
        img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# ---------------------------
# Predict Food (Filtered to CSV Foods)
# ---------------------------
def predict_food(img, top_k=5):
    """
    Predict food name & probability from image.
    Only returns items that match the food list from calorie_data.csv.
    """
    x = preprocess_image(img)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=top_k)[0]

    # Loop through predictions and find first match in food_list
    for (_, label, prob) in decoded:
        label_clean = label.replace("_", " ").lower()
        if label_clean in food_list:
            return label_clean, prob * 100

    # If no match, return top prediction anyway
    return decoded[0][1].replace("_", " ").lower(), decoded[0][2] * 100

# ---------------------------
# Get Calories
def normalize_name(name):
    return name.lower().replace("_", "").replace(" ", "")

def get_calories(food_name):
    """Return calorie info if food is in calorie_data.csv"""
    normalized_food = normalize_name(food_name)
    calorie_df['normalized'] = calorie_df['food'].apply(normalize_name)

    row = calorie_df[calorie_df['normalized'] == normalized_food]
    if not row.empty:
        return int(row['calories'].values[0])
    return None

