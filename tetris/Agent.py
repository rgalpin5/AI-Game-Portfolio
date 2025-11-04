import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import json
import os
import glob



# Learning params
MAX_MEMORY = 100_000
BATCH_SIZE = 1024
LR = 0.001
GAMMA = 0.995
EPSILON_START = 1.0


class Linear_QNet(nn.Module):
    """ Linear neural network for Q-learning"""
    def __init__(self, input_size, hidden_size, output_size):
        """Initializes the layers for the Neural Network.

        Args:
            input_size (int): number of features in the input layer
            hidden_size (int): number of neurons in the hidden layer
            output_size (int): number of values in the output layer
        """
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        """Forward pass of the Neural Network.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor of Q-values
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        # Output
        x = self.linear3(x)
        return x

class Agent():
    """The reinforcement learning agent that plays tetris"""
    def __init__(self):
        """Initializes the agent, memory, optimizer and model"""
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.gamma = GAMMA

        # Initialize model and memory
        self.model = Linear_QNet(6, 128, 1)
        self.target_model = Linear_QNet(6, 128, 1)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.memory = deque(maxlen=MAX_MEMORY)

        # Optimizer for adjusting weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        # Mean Squared Error loss function
        self.loss_fn = nn.MSELoss()

        self.state_file = 'runs/agent_state.json'
        self.load_model()
        self.load_state()

    def load_model(self, file_name=None):
        """Loads model weights from .pth file.

        If no file_name is provided, loads most recently modified file

        Args:
            file_name (str, optional): The path to the model file. Defaults to None.
        """
        model_folder_path = 'models'

        if file_name is None:
            if not os.path.exists(model_folder_path):
                print("No 'models' directory found. Starting with a new model.")
                return

            # Find all files matching the checkpoint pattern
            checkpoints = glob.glob(os.path.join(model_folder_path, 'tetris_game_*.pth'))

            if not checkpoints:
                print("No checkpoints found. Starting with a new model.")
                return

            # Find the most recently modified file
            try:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                file_name = latest_checkpoint
                print(f"Found latest checkpoint: {file_name}")
            except Exception as e:
                print(f"Error finding latest checkpoint: {e}. Starting new model.")
                return

        if os.path.exists(file_name):
            try:
                self.model.load_state_dict(torch.load(file_name))
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Loaded existing model: {file_name}")
            except Exception as e:
                print(f"Error loading model {file_name}: {e}")
        else:
            # This case should only be hit if a specific file_name was passed but didn't exist.
            print(f"No model file found at {file_name}. Starting with a new model.")
    def load_state(self):
        """Loads the agents state from a JSON file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.n_games = data['n_games']
                    self.epsilon = data['epsilon']
                    print(f"Loaded agent state: Game {self.n_games}, Epsilon {self.epsilon:.4f}")
            except Exception as e:
                # Couldn't load state file
                print(f"Error loading state file {self.state_file}: {e}. Starting fresh.")
        else:
            # No file exists
            print(f"No state file found at {self.state_file}. Starting fresh.")

    def save_state(self):
        """Saves the agent state to a JSON file."""
        try:
            data = {'n_games': self.n_games, 'epsilon': self.epsilon}
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving state: {e}")

    def save_model(self, file_name='checkpoint_model.pth'):
        """Saves the current models weights to a .pth file.

        Args:
            file_name (str, optional): The path to the model file. Defaults to 'checkpoint_model.pth'.
        """
        # Create directory if it doesn't exist
        model_folder_path = 'models'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        try:
            torch.save(self.model.state_dict(), file_path)
            print(f"Saved model checkpoint to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def remember(self, state_features, reward, next_state_features_list, done):
        """Stores an experience tuple in memory.

        Args:
            state_features (torch.Tensor): Tensor with features of the current state
            reward (torch.Tensor): Reward received after taking the action
            next_state_features_list (list): List of next state features
            done (bool): Flag indicating if the experience is done
        """
        self.memory.append((state_features, reward, next_state_features_list, done))

    def get_action(self, placement_features_list, epsilon):
        """Selects an action using the epsilon-greedy policy.

        With probability epsilon, a random action is chosen.
        Otherwise, the action with the highest predicted Q is chosen.

        Args:
            placement_features_list (list): A list of feature tensors
            epsilon (float): The probability of taking a random action

        Returns:
            tuple: A tuple containing the action type ('place'), the index of the
                chosen move, the features of the chosen move, and its
                predicted Q-value.
        """
        self.epsilon = epsilon


        placement_batch = torch.cat(placement_features_list, dim=0)
        # Get Q-Value predictions for all placements without gradient calc during inference
        with torch.no_grad():
            placement_q_values = self.model(placement_batch)

        # Get highest Q action
        best_placement_q, best_placement_idx = torch.max(placement_q_values, dim=0)

        # Convert to python number
        best_placement_idx = best_placement_idx.item()



        chosen_move_idx = best_placement_idx
        # Epsilon-greedy, "Explore" if random number is less than epsilon
        if random.random() < self.epsilon:
            chosen_move_idx = random.randint(0, len(placement_features_list) - 1)

        # Get predicted Q val of chosen move and the feature set
        q_pred = placement_q_values[chosen_move_idx]
        chosen_features = placement_features_list[chosen_move_idx]

        # Returns action info and the Q_Value
        return 'place', chosen_move_idx, chosen_features, q_pred.unsqueeze(0)

    def train_long_memory(self):
        """Trains the model on a batch of experiences.

        Samples a mini-batch from memory, calculates Q-values using the Bellman equation,
        and performs a gradient descent step to update the model weights.

        Returns:
            float or None: The loss value for the training step, or None if memory is not large enough
        """
        # Only train when memory has enough samples
        if len(self.memory) < BATCH_SIZE:
            return

        mini_batch = random.sample(self.memory, BATCH_SIZE)

        # Get all current state features, rewards, 'next possible state'features and all 'done' flags
        state_features_batch = torch.cat([item[0] for item in mini_batch], dim=0)
        reward_batch =  torch.tensor([item[1] for item in mini_batch], dtype=torch.float32)
        next_features_lists = [item[2] for item in mini_batch]
        done_batch = torch.tensor([item[3] for item in mini_batch], dtype=torch.float32)

        # Q Val predictions for current states
        Q_pred = self.model(state_features_batch)

        # Store max Q-Val for each 'next state'
        Q_next_max = []
        # Iterate through each samples next moves
        for features_list in next_features_lists:
            if not features_list:
                Q_next_max.append(0.0)
            else:
                # Stack next moves for this sample
                next_batch = torch.cat(features_list, dim=0)
                with torch.no_grad():
                    next_q_values = self.target_model(next_batch)
                # Find best Q val for next state
                Q_next_max.append(torch.max(next_q_values).item())
        # Convert list to tensor
        Q_next_max_tensor = torch.tensor(Q_next_max, dtype=torch.float32)

        # Bellman eq for Q
        Q_target = reward_batch + (self.gamma * Q_next_max_tensor * (1 - done_batch))


        # Training step
        self.optimizer.zero_grad()
        loss =  self.loss_fn(Q_pred, Q_target.unsqueeze(1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        """Updates the target network's weights to match the main model's."""
        self.target_model.load_state_dict(self.model.state_dict())
