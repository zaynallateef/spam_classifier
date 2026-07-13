

import os

import requests
from flask import Flask, render_template, request

app = Flask(__name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    message_text = ""

    if request.method == "POST":
        message_text = request.form.get("message", "").strip()

        if not message_text:
            error = "Please enter a message to classify."
        else:
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/predict",
                    json={"text": message_text},
                    timeout=10,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.RequestException as e:
                error = f"Could not reach the classifier API: {e}"

    return render_template(
        "index.html", result=result, error=error, message_text=message_text
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
