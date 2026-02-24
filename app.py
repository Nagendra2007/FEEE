@app.route("/analyze", methods=["POST"])
def analyze():
    global latest_result, latest_timestamp

    # Check if any file exists
    if len(request.files) == 0:
        print("No files received")
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Get first file regardless of field name
        file = list(request.files.values())[0]

        image = Image.open(file.stream)
        image = image.convert("RGB")
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

        print("Gemini processing time:", time.time() - start_time)

        latest_result = result.text
        latest_timestamp = time.time()

        return jsonify({"text": latest_result})

    except Exception as e:
        print("Processing error:", str(e))
        return jsonify({"error": "Processing failed"}), 500
