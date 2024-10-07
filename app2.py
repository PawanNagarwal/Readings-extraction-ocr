import os
import base64
# import requests
import streamlit as st
from openai import OpenAI

api_key = os.environ['OPENAI_API_KEY']         
client = OpenAI(api_key=api_key)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

img_prompt = """you will be given the image of glucometer reading of a patient, give me the blood glucose concentration from that image 
and use this format to present the output in json :-     
{                             
    "blood glucose concentration": 
}             
"""        

def get_readings(image_path):
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": img_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            },
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("OCR Readings Extraction for Glucometer")

uploaded_file = st.file_uploader("Upload an image....", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp_image.jpeg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Extracting readings...")
    
    try:
        readings = get_readings("temp_image.jpeg")
        st.write("Extracted Readings:")
        st.write(readings)
    except Exception as e:
        st.write("Error in processing the image.")
        st.write(e)
