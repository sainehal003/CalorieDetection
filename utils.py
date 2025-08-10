import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import pandas as pd

# Build the model from MobileNetV2 base
def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # freeze base model for inference

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(5, activation='softmax')  # 5 classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build model (no loading .h5)
model = build_model()

# Optional: load your trained weights if you have them saved separately
# model.load_weights("model/food_classifier_weights.h5")

# Load calorie data
calorie_df = pd.read_csv("data/calorie_data.csv")

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_food(img):
    processed = preprocess_image(img)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction)
    return predicted_class

def get_food_label(class_id):
    class_labels = ['pizza', 'burger', 'salad', 'sushi', 'fried_rice']
    return class_labels[class_id]

def get_calories(food_name):
    row = calorie_df[calorie_df['food'] == food_name]
    if not row.empty:
        return int(row['calories'].values[0])
    return None
