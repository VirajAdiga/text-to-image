import os
import io
import time

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

prompts = [""]

API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


for i in range(len(prompts)):
    image_bytes = query({
        "inputs": prompts[i],
    })

    image = Image.open(io.BytesIO(image_bytes))
    image.save(f"output_image_{i}.png")
    time.sleep(3)
