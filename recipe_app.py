import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 🎯 Load dataset
df = pd.read_excel("Magical Recipe detailed.xlsx)

# 🎯 Encode categorical values
mood_encoder = LabelEncoder()
weather_encoder = LabelEncoder()
recipe_encoder = LabelEncoder()

df['Mood_enc'] = mood_encoder.fit_transform(df['Mood'])
df['Weather_enc'] = weather_encoder.fit_transform(df['Weather'])
df['Recipe_enc'] = recipe_encoder.fit_transform(df['Recipe'])

# 🎯 Train the model
X = df[['Mood_enc', 'Weather_enc']]
y = df['Recipe_enc']

model = DecisionTreeClassifier()
model.fit(X, y)

# 🎨 Streamlit UI
st.set_page_config(page_title="Recipe Recommender", page_icon="🍪")
st.title("Recipe Recommender")

# Mood and Weather dropdowns
mood = st.selectbox("How are you feeling?", options=df['Mood'].unique())
weather = st.selectbox("What's the weather like?", options=df['Weather'].unique())

# Predict button
if st.button(" Recommend Recipe"):
    try:
        # Encode input
        mood_enc = mood_encoder.transform([mood])[0]
        weather_enc = weather_encoder.transform([weather])[0]

        # Predict
        prediction_enc = model.predict([[mood_enc, weather_enc]])[0]
        prediction = recipe_encoder.inverse_transform([prediction_enc])[0]

        # Get steps
        if 'Recipe_Description' in df.columns:
            steps = df[df['Recipe'] == prediction]['Recipe_Description'].values[0]
        else:
            steps = "No instructions available."

        # Display output
        st.success(f" Recommended Recipe: **{prediction}**")
        st.info(f"📋 Steps:\n{steps}")

    except:
        st.error("⚠️ Error: Invalid mood or weather. Please choose from dropdown.")
