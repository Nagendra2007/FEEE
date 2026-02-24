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

    # Check if any file was uploaded
    if len(request.files) == 0:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Get first uploaded file (Kodular sends unknown field name)
        file = list(request.files.values())[0]

        image = Image.open(file.stream)
        image = image.convert("RGB")

        # Resize for speed
        image.thumbnail((512, 512))

        start_time = time.time()

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
        print("Error:", str(e))
        return jsonify({"error": "AI processing failed"}), 500


# ===== LATEST ROUTE (FOR AUTO MODE LATER) =====
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
