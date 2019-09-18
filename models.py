import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixFactorization(torch.nn.Module):
    def __init__(self, num_users, num_movies, n_factors=20):
        super().__init__()
        # create user embeddings
        self.user_emb = torch.nn.Embedding(num_users, n_factors,sparse=True)
        # create item embeddings
        self.movie_emb = torch.nn.Embedding(num_movies, n_factors,sparse=True)

    def forward(self, x):
        user = x[0]
        movie = x[1]
        # matrix multiplication
        return (self.user_emb(user)*self.movie_emb(movie)).sum(1)

    
class DenseNet(nn.Module):
    def __init__(self, num_users, num_movies, H1=128, D_out=1, n_factors=20):
        super().__init__()
        # user and item embedding layers
        self.user_emb = torch.nn.Embedding(num_users, n_factors, sparse=True)
        self.movie_emb = torch.nn.Embedding(num_movies, n_factors, sparse=True)
        # linear layers
        self.linear1 = torch.nn.Linear(n_factors*2, H1)
        self.linear2 = torch.nn.Linear(H1, D_out)

    def forward(self, x):
        users = x[0]
        movies = x[1]
        users_embedding = self.user_emb(users)
        movies_embedding = self.movie_emb(movies)        
        # concatenate user and item embeddings to form input
        x = torch.cat([users_embedding, movies_embedding], dim=1)                
        h1_relu = F.relu(self.linear1(x))
        output_scores = self.linear2(h1_relu).squeeze(-1)        
        return output_scores
