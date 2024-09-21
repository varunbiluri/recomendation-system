from flask import Flask, jsonify, request
import torch
import torch.nn as nn

app = Flask(__name__)

# Define your model class (must match the structure you used during training)
class RecommendationModel(nn.Module):
    def __init__(self):
        super(RecommendationModel, self).__init__()
        # Define your model layers (as done in the trained model)
        self.fc = nn.Linear(10, 5)  # Example: Adjust to your model's architecture

    def forward(self, x):
        return self.fc(x)

# Load the trained model
model = RecommendationModel()
model.load_state_dict(torch.load('recommendation_model.pth', map_location=torch.device('cpu')))  # Load the model onto CPU
model.eval()  # Set the model to evaluation mode

# Route to provide recommendations based on user_id
@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    try:
        # Example data based on user_id (modify this part based on your dataset or input needs)
        user_input = get_user_data(user_id)  # Replace with actual logic to retrieve user input
        
        if user_input is None:
            return jsonify({"error": f"No data found for user {user_id}"}), 404

        # Convert the input to a PyTorch tensor (adjust shape as per your model's input requirements)
        input_tensor = torch.tensor(user_input, dtype=torch.float32)

        # Get recommendations from the model
        with torch.no_grad():  # Disable gradient computation for efficiency
            recommendations = model(input_tensor.unsqueeze(0))  # Unsqueeze if batch size is 1

        # Convert the tensor output to a Python list
        recommendations_list = recommendations.squeeze().tolist()

        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations_list
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Helper function to retrieve user data based on user_id
# You can modify this function to fit your data handling logic
def get_user_data(user_id):
    # Dummy example: Replace with logic to get user input
    # For instance, this could be features like user behavior, demographics, etc.
    user_data = {
        1: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        2: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4],
        # Add more user data as needed
    }
    return user_data.get(user_id, None)

if __name__ == '__main__':
    app.run(debug=True)
