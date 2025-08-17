import streamlit as st
from PIL import Image
from utils import predict_food, get_calories

st.set_page_config(page_title="Food Calorie Estimator", layout="centered")

st.title("🍽️ Food Recognition & Calorie Estimation")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image_file = Image.open(uploaded_file)
    st.image(image_file, caption='Uploaded Food Image', use_column_width=True)

    with st.spinner("Predicting..."):
        food_name, confidence = predict_food(image_file)
        calories = get_calories(food_name)

    st.success(f"🍕 Predicted Food: **{food_name.capitalize()}** ({confidence:.2f}%)")
    if calories:
        st.info(f"🔥 Estimated Calories: **{calories} kcal** per 100g")
    else:
        st.warning("No calorie info found for this food.")