from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
from game import decide_winner, get_computer_move
import os

app = Flask(__name__)
model = tf.keras.models.load_model('rps_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Inside Predict Function")
        data = request.get_json()
        
        # Get base64 encoded string from POST request and convert to image
        encoded_image = data['image']
        print("Encoded Image Received")
        
        decoded_image = base64.b64decode(encoded_image.split(",")[1])
        image = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), cv2.IMREAD_COLOR)

        # Preprocess the image
        img = cv2.resize(image, (150, 150))
        img = tf.cast(img, tf.float32) / 255.0

        # Predict user's move
        prediction = model.predict(np.expand_dims(img, axis=0))
        print("Prediction Done")

        predicted_label = np.argmax(prediction, axis=1)
        moves = ["rock", "paper", "scissors"]
        user_move = moves[predicted_label[0]]

        # Get computer's move
        computer_move = get_computer_move()

        # Decide the winner
        result = decide_winner(user_move, computer_move)

        return jsonify(user_move=user_move, computer_move=computer_move, result=result)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(debug=True)
