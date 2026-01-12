import streamlit as st

# 🛑 SET PAGE CONFIG MUST COME FIRST
st.set_page_config(page_title="Image to Audio Description", layout="centered")

from PIL import Image
import requests
from gtts import gTTS
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Your Groq API key here
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# Load BLIP model
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

# Function to query Groq API
# Function to query Groq API — corrected endpoint + model
def query_groq(prompt, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"   # <-- use /openai/v1/...
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-8b-instant",   # <-- valid Groq model
        "messages": [
            {"role": "system", "content": "You are an expert in understanding and describing images."},
            {"role": "user", "content": f"Refine this image description: {prompt}"}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Groq API request failed: {str(e)}"


# Generate audio from text
def generate_audio(text):
    tts = gTTS(text)
    audio_path = "image_description.mp3"
    tts.save(audio_path)
    return audio_path

# App Title
st.title("🖼️🎤 Image to Audio Description with Groq")

# Upload image
uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Audio Description"):
        # Generate caption with BLIP
        st.info("Generating image caption...")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        st.success(f"BLIP Caption: {caption}")

        # Refine using Groq
        st.info("Refining caption using Groq...")
        refined_description = query_groq(caption, GROQ_API_KEY)

        if refined_description.lower().startswith("groq api request failed"):
            st.error(refined_description)
        else:
            st.subheader("📝 Refined Description:")
            st.write(refined_description)

            # Convert to audio
            st.info("Converting description to audio...")
            audio_path = generate_audio(refined_description)
            audio_file = open(audio_path, "rb").read()
            st.audio(audio_file, format="audio/mp3")
