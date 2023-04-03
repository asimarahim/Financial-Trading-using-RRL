import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# Define a neural network model for the Reinforcement Learning framework
class RRLModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        # Define a linear layer with input size m+1 and output size 1
        self.neuron = nn.Linear(m+1, 1, bias=True)
        # Initialize the weights of the linear layer uniformly between -0.2 and 0.2, and the bias to 0
        nn.init.uniform_(self.neuron.weight, -.2, .2)
        nn.init.constant_(self.neuron.bias, 0)

    def forward(self, features):
        # Apply a hyperbolic tangent activation function to the output of the linear layer and return the result
        return torch.tanh(
            self.neuron(features)
        )


# Define a function to compute the Sharpe Ratio of a set of returns
def sharpe_ratio(returns: torch.Tensor, eps: float = 1e-6):
    expected_return = torch.mean(returns, dim=-1)
    # The reference writeup used the biased STD estimator
    expected_squared_return = torch.mean(returns ** 2, dim=-1)
    sharpe = expected_return / (torch.sqrt(
        expected_squared_return - expected_return ** 2
    ) + eps)
    return sharpe


# Define a function to compute the reward for a given set of asset returns, trading parameters, and previous portfolio weights
def reward_function(asset_returns: torch.Tensor, miu: float, delta: float, Ft: torch.Tensor, m: int):
    # Compute the number of time periods for which we have asset returns
    n = Ft.shape[-1] - 1
    # Compute the returns for the trading period
    returns = miu * (
        Ft[:n] * asset_returns[m:m+n]
    ) - (
        delta * torch.abs(Ft[1:] - Ft[:n])
    )
    # Compute the Sharpe Ratio of the returns
    sharpe = sharpe_ratio(returns)
    return returns, sharpe


# Define a function to update the portfolio weights using the RRL model
def update_Ft(normalized_asset_returns: torch.Tensor, model: RRLModel):
    # Get the number of past asset returns to use for prediction
    m = model.m
    # Get the number of time periods for which we have asset returns
    t = normalized_asset_returns.shape[-1] - m
    # Initialize an array to hold the portfolio weights
    Ft = torch.zeros(t + 1).to(normalized_asset_returns.device)
    for i in range(1, t):
        # Construct a feature vector consisting of the past m normalized asset returns and the previous portfolio weight
        features = torch.cat([
            normalized_asset_returns[i-1:i+m-1], Ft[i-1:i]
        ])
        # Predict the next portfolio weight using the RRL model
        Ft[i] = model(features)
    return Ft[1:]


def gradient_accent(asset_returns: torch.Tensor,
                     normalized_asset_returns: torch.Tensor,
                     model: RRLModel,
                     max_iter: int, lr: float):

    # Create an optimizer for the model using stochastic gradient descent (SGD)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Create an empty list to store the rewards obtained during optimization
    rewards = []
    
    # Iterate over the maximum number of iterations specified
    for i in range(max_iter):
        # Zero out the gradients in the optimizer
        optimizer.zero_grad()
        
        # Update Ft using the current model
        Ft = update_Ft(normalized_asset_returns, model)
        
        # Calculate the returns and reward for the current model
        returns, reward = reward_function(asset_returns, miu=1., delta=0, Ft=Ft, m=model.m)
        
        # Calculate the gradients for the reward function and perform a backward pass
        # (-1 * reward) is used because we are performing gradient *ascent*
        (-1 * reward).backward()
        
        # Take an optimizer step to update the model parameters
        optimizer.step()
        
        # Append the current reward to the rewards list
        rewards.append(reward.detach().cpu())
    
    # Return the rewards, returns, and Ft values as a tuple
    return rewards, returns, Ft

def train(prices: torch.Tensor, m: int, t: int, delta: float = 0, max_iter: int = 100, lr: float = 0.1):
    # Check that the input tensor is 1-dimensional
    assert len(prices.size()) == 1
    # Compute the asset returns as the ratio of the change in price to the previous price
    asset_returns = (prices[1:] - prices[:-1]).float() / prices[:-1]
    # Use a StandardScaler to normalize the first m+t asset returns
    scaler = StandardScaler()
    normalized_asset_returns = torch.tensor(scaler.fit_transform(
        asset_returns[:m+t][:, None].numpy()
    )[:, 0]).float()
    # Initialize the RRL model with m and perform gradient ascent to train the weights
    model = RRLModel(m)
    train_rewards, train_returns, train_Ft = gradient_accent(
        asset_returns, normalized_asset_returns, model, max_iter, lr
    )
    # Use the trained model to predict the next t steps of asset returns
    normalized_asset_returns = torch.tensor(
        scaler.transform(asset_returns[t:][:, None].numpy())[:, 0]
    ).float()
    Ft_ahead = update_Ft(normalized_asset_returns, model)
    returns_ahead, reward_ahead = reward_function(asset_returns[t:], 1., delta, Ft_ahead, model.m)
    # Compute the cumulative percentage returns for both the asset returns and the RRL model returns
    percentage_returns = (torch.exp(
        torch.log(1 + returns_ahead).cumsum(dim=-1)
    ) - 1) * 100
    return {
        "valid_reward": reward_ahead,
        "valid_Ft": Ft_ahead,
        "valid_asset_returns": asset_returns[m+t:],
        "valid_asset_percentage_returns": (torch.exp(
            torch.log(1 + asset_returns[m+t:]).cumsum(dim=-1)
        ) - 1) * 100,
        "valid_percentage_returns": percentage_returns,
        "rewards_iter": train_rewards,
        "train_percentage_returns": (torch.exp(
            torch.log(1 + train_returns).cumsum(dim=-1)
        ) - 1) * 100,
        "train_Ft": train_Ft
    }
