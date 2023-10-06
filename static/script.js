document.addEventListener('DOMContentLoaded', () => {
    let form = document.querySelector("form");
    let userMoveSpan = document.querySelector("#user-move");
    let computerMoveSpan = document.querySelector("#computer-move");
    let resultSpan = document.querySelector("#result");

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        let formData = new FormData(form);
        
        // Send the image to the server for prediction
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Display the prediction results
            userMoveSpan.textContent = data.user_move;
            computerMoveSpan.textContent = data.computer_move;
            resultSpan.textContent = data.result;
        })
        .catch(error => {
            console.error('Error:', error);
            resultSpan.innerHTML = "An error occurred. Please try again.";
        });
    });
});
