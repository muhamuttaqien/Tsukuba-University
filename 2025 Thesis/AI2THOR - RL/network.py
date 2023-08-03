import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Define DQN architecture."""
    
    def __init__(self, width, height, action_size):
        """Initialize parameters and build model."""
            
        super(DQN, self).__init__()
        
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
        
        # this will output Q(s,left) and Q(s,right) (where s is the input to the network) [left0exp,right0exp]
        Qsa = self.head(state.view(state.size(0), -1))
        
        return Qsa
    
    