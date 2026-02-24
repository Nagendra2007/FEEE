from flask import Flask, request, jsonify
from google import genai
from PIL import Image
import os
import time

app = Flask(__name__)

# ===== GLOBAL STORAGE =====
latest_result = ""
latest_timestamp = 0

# ===== GEMINI CLIENT =====
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

# ===== ANALYZE ROUTE =====
@app.route("/analyze", methods=["POST"])
def analyze():
    global latest_result, latest_timestamp

    try:
        # Check if any file was uploaded
        if len(request.files) == 0:
            print("No files received")
            return jsonify({"error": "No image uploaded"}), 400

        # Accept first uploaded file (Kodular sends unknown key)
        file = list(request.files.values())[0]

        print("File received:", file.filename)

        # Open and process image
        image = Image.open(file.stream)
        image = image.convert("RGB")
        image.thumbnail((512, 512))  # Resize for speed

        start_time = time.time()

        # Gemini request
        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Describe main objects briefly.",
                image
            ],
            generation_config={
                "max_output_tokens": 40
            }
        )

        processing_time = time.time() - start_time
        print("Gemini processing time:", processing_time)

        latest_result = result.text
        latest_timestamp = time.time()

        return jsonify({"text": latest_result})

    except Exception as e:
        print("Processing error:", str(e))
        return jsonify({"error": "Processing failed"}), 500


# ===== LATEST ROUTE =====
@app.route("/latest", methods=["GET"])
def latest():
    return jsonify({
        "text": latest_result,
        "timestamp": latest_timestamp
    })


# ===== RUN SERVER =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
