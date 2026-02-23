from flask import Flask, request, jsonify
import requests
from google import genai
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)

# Put your Gemini API key here
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    image_url = data.get("image_url")

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    # Download image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Send to Gemini
    result = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "Describe this image briefly in two lines for a blind person.Only give the information",
            image
        ]
    )

    return jsonify({"text": result.text})

if __name__ == "__main__":
  port = int(os.environ.get("PORT",5000))

  app.run(host="0.0.0.0", port=port)


