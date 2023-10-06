import random
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('rps_model.h5')


def get_user_input():
    # Capture video from webcam
    cap = cv2.VideoCapture(0)
    
    print("Show your move to the camera! Press 's' to take the shot!")
    while True:
        ret, frame = cap.read()
        cv2.imshow('RPS Game', frame)
        
        # Save the image when 's' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('user_move.jpg', frame)
            break
            
    cap.release()
    cv2.destroyAllWindows()

    # Load and preprocess the image
    img = cv2.imread('user_move.jpg')
    img = cv2.resize(img, (150, 150))
    img = tf.cast(img, tf.float32) / 255.0

    # Expand dimensions for the model and predict
    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_label = np.argmax(prediction, axis=1)

    # Map the predicted label to its string representation
    moves = ["rock", "paper", "scissors"]
    return moves[predicted_label[0]]

def decide_winner(user, computer):
    if user == computer:
        return "Draw"
    if ((user == "rock" and computer == "scissors") or 
        (user == "scissors" and computer == "paper") or 
        (user == "paper" and computer == "rock")):
        return "User Wins"
    return "Computer Wins"


def get_computer_move():
    moves = ["rock", "paper", "scissors"]
    return random.choice(moves)

def play_game():
    user_move = get_user_input()
    computer_move = get_computer_move()
    
    print(f"Your Move: {user_move}")
    print(f"Computer's Move: {computer_move}")
    
    result = decide_winner(user_move, computer_move)
    print(result)

