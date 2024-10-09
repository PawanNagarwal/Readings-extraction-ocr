import cv2
import os
from datetime import datetime
import os
import base64
# import requests
import streamlit as st
from openai import OpenAI


os.environ['OPENAI_API_KEY'] = "api_key" 
client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

img_prompt = """you will be given the image of glucometer reading of a patient, give me the blood glucose concentration from that image 
and use this format to present the output in json :-     
{                             
    "blood glucose concentration": 
}             
"""       

def capture_and_save_image():
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not capture image.")
        cap.release()
        return None, None

    # Release the camera
    cap.release()

    # Create a directory to store the image if it doesn't exist
    directory = "captured_images"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Generate a unique filename using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"captured_image_{timestamp}.jpg"
    filepath = os.path.join(directory, filename)

    # Save the image
    cv2.imwrite(filepath, frame)

    base64_image = encode_image(filepath)
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
    result = response.choices[0].message.content
    return result, filepath


st.title("OCR Readings extractor")
if st.button("capture and analyze"):
    with st.spinner("capturing and anlayzing image..."):
        result, image_path = capture_and_save_image()
        if result and image_path:
            st.success("Image captured and analyzed successfully")    
            st.image(image_path, caption="Captured image", use_column_width=True)
            st.write(result) 
            # st.json(result)
        else:
            st.error("failed to capture or analyze image.")

st.markdown("---")
st.write("Note: this app requires camera access to capture the glucometer reading")