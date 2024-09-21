import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load the datasets
movies = pd.read_csv('D:/job preparation/Collabration/dataset/movie.csv')
ratings = pd.read_csv('D:/job preparation/Collabration/dataset/rating.csv')
print("Loaded data successfully")

# Number of unique users and movies
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()

# Reindex user and movie IDs to ensure continuous indexing
user_ids = ratings['userId'].astype('category').cat.codes
movie_ids = ratings['movieId'].astype('category').cat.codes

# Convert to PyTorch tensors
user_tensor = torch.tensor(user_ids.values, dtype=torch.long)
movie_tensor = torch.tensor(movie_ids.values, dtype=torch.long)
rating_tensor = torch.tensor(ratings['rating'].values, dtype=torch.float)

# Create dataset and dataloader for mini-batch training
dataset = TensorDataset(user_tensor, movie_tensor, rating_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the Matrix Factorization model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

    def forward(self, user, movie):
        user_embedded = self.user_embedding(user)
        movie_embedded = self.movie_embedding(movie)
        return (user_embedded * movie_embedded).sum(1)

# Set a higher embedding dimension for better user-movie interaction representation
embedding_dim = 50
model = MatrixFactorization(num_users, num_movies, embedding_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with mini-batch training
epochs = 5
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    print("Model created successfully")
    running_loss = 0.0

    for user_batch, movie_batch, rating_batch in dataloader:
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        output = model(user_batch, movie_batch)

        # Compute the loss
        loss = criterion(output, rating_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the loss for each epoch
    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader)}')

# Save the model after training
torch.save(model.state_dict(), 'matrix_factorization_model.pth')
