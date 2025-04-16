from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# Load model and tokenizer
def load_model():
    repo_id = "fazalahmad/Bert"  # Make sure this is public or you handle auth if private
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return model, tokenizer

model, tokenizer = load_model()

# Route for the web page
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()

    # Adjust labels to match your model's class order
    labels = ["Bad", "Good", "Neutral"]
    prediction = labels[predicted_class_id]

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
