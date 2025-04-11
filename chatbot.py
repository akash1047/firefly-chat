import tensorflow as tf
import numpy as np
import json
import pickle
import random
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request, jsonify


model = tf.keras.models.load_model("chatbot_model.h5")


with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)
with open("label_encoder.pkl", "rb") as handle:
    le = pickle.load(handle)


with open("intents.json") as content:
    data = json.load(content)

responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}
input_shape = model.input_shape[1]

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get", methods=["GET"])
def chatbot_response():
    user_text = request.args.get("msg")

    user_text = "".join(
        [letters.lower() for letters in user_text if letters not in string.punctuation]
    )
    input_sequence = pad_sequences(
        [tokenizer.texts_to_sequences([user_text])[0]], maxlen=input_shape
    )

    output = model.predict(input_sequence)
    response_tag = le.inverse_transform([output.argmax()])[0]
    bot_response = random.choice(responses[response_tag])

    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
