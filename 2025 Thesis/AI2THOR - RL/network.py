import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Define DQN architecture."""
    
    def __init__(self, width, height, action_size, seed):
        """Initialize parameters and build model."""
            
        super(DQN, self).__init__()
        
        self.seed = torch.manual_seed(seed)

        # CNN will take in the difference between the current and previous screen patches
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        # number of linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it
        def conv2d_size_outputs(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_outputs(conv2d_size_outputs(conv2d_size_outputs(width)))
        convh = conv2d_size_outputs(conv2d_size_outputs(conv2d_size_outputs(height)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, action_size)
        
    # the network will be inputted by state
    def forward(self, state):
        """Build a network that maps state into action values."""
        
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        
        Qsa = self.head(state.reshape(state.size(0), -1))
        
        return Qsa


class PolicyNetwork(nn.Module):
    """Policy Network architecture with Convolutional Layers."""
    
    def __init__(self, width, height, hidden_size, output_size, seed):
        
        super(PolicyNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the flattened output from convolutional layers
        convw = width // 4  # Assuming 2 max-pooling layers with kernel_size=2 and stride=2
        convh = height // 4
        linear_input_size = convw * convh * 32
        
        # Fully connected layers
        linear_input_size = 320000
        
        self.fc1_layer = nn.Linear(linear_input_size, hidden_size)
        self.fc2_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, state):
        
        # Convolutional layers
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        
        # Flatten the output before passing it through fully connected layers
        state = state.view(state.size(0), -1)
        
        # Fully connected layers
        logits = F.relu(self.fc1_layer(state))
        logits = self.fc2_layer(logits)
        probs = F.softmax(logits, dim=1) # Assuming dim=1 is the correct dimension for softmax
        
        return probs


class ValueNetwork(nn.Module):
    """Value Network architecture with Convolutional Layers."""
    
    def __init__(self, width, height, hidden_size, output_size, seed):
        
        super(ValueNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the flattened output from convolutional layers
        convw = width // 4  # Assuming 2 max-pooling layers with kernel_size=2 and stride=2
        convh = height // 4
        linear_input_size = convw * convh * 32
        
        # Fully connected layers
        linear_input_size = 320000
        
        self.fc1_layer = nn.Linear(linear_input_size, hidden_size)
        self.fc2_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, state):
        
        # Convolutional layers
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        
        # Flatten the output before passing it through fully connected layers
        state = state.view(state.size(0), -1)
        
        # Fully connected layers
        state_value = F.relu(self.fc1_layer(state))
        state_value = self.fc2_layer(state_value)
        
        return state_value
