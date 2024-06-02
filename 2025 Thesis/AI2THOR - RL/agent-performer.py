import os
import time
import random
import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from PIL import Image
from IPython.display import clear_output

import ai2thor
import ai2thor_colab
from ai2thor_colab import plot_frames
from ai2thor.controller import Controller

from ai2thor.platform import CloudRendering
controller = Controller(platform=CloudRendering)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.distributions import Categorical

from network import DQN
from utils import to_torchdim, frame2tensor, encode_feedback


import warnings
warnings.filterwarnings('ignore')

import argparse

# Create an argument parser
parser = argparse.ArgumentParser()

# Define the expected arguments as key-value pairs
parser.add_argument('--agent', type=str, default='./agents/AI2THOR_MM_RL_3OBJ_R20.pth')
parser.add_argument('--floor', type=str, default='20')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values using the argument names
arg1 = args.agent
arg2 = args.floor

# ## Set Environment
print("(1/5) Init the environment...")

floor_index = random.randint(0, 30)
floor_index = arg2

controller = Controller(
    agentMode = "default", # arm
    visibilityDistance = 0.50,
    scene = f"FloorPlan{floor_index}",

    # step sizes
    snapToGrid = True,
    gridSize = 0.25,
    rotateStepDegrees = 90,

    # image modalities
    renderInstanceSegmentation = False,
    renderDepthImage = False,
    renderSemanticSegmentation = False,
    renderNormalsImage = False,
    
    # camera properties
    width = 900,
    height = 600,
    fieldOfView = 120,
    
    # set seed for reproducability
    # seed = 90,
)


# ## Set Configs

is_cuda = torch.cuda.is_available()

if is_cuda: device = torch.device('cuda')
else: device = torch.device('cpu')

NUM_EPISODES = 10000 # 6000

SCREEN_WIDTH = SCREEN_HEIGHT = 100

action_space = ["MoveAhead", "RotateLeft", "RotateRight"]


# ## Load Word2Vec

print("(2/5) Load Word2Vec...")

from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'weights/glove.6B.100d.txt'
word2vec_output_file = 'weights/glove.6B.100d.txt.word2vec'

glove2word2vec(glove_input_file, word2vec_output_file)
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

pretrained_embeddings = torch.FloatTensor(word2vec_model.vectors)


# ## Build Model

class VisualModel(nn.Module):
    
    def __init__(self, seed):
        
        super(VisualModel, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        
    def forward(self, x):
        
        return self.cnn(x)
    
    
class TextModel(nn.Module):
    
    def __init__(self, pretrained_embedding, hidden_dim, seed):
        
        super(TextModel, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=True)
        self.rnn = nn.LSTM(pretrained_embedding.shape[1], hidden_dim)
        
    
    def forward(self, x):
        
        x = x.long()
        
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = output.view(output.shape[0], -1)
        
        return output

class MultimodalDQN(nn.Module):
    
    def __init__(self, visual_model, text_model, action_size, seed):
        
        super(MultimodalDQN, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.visual_model = visual_model
        self.text_model = text_model
        
        # Define three fully connected layers
        self.fc1 = nn.Linear(640, 512) # 256, 512, 1024
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        
    def forward(self, visual_input, text_input):
        
        visual_features = self.visual_model(visual_input.to(device))
        text_features = self.text_model(text_input.to(device))
        
        # Concatenate visual and text features
        combined_features = torch.cat((visual_features, text_features), dim=1)
        
        # Apply fully connected layers
        combined_features = F.relu(self.fc1(combined_features))
        combined_features = F.relu(self.fc2(combined_features))
        q_values = self.fc3(combined_features)
        
        return q_values

# ## Build Agent

class DQNAgent():
    """The agent interacting with and learning from the environment."""
    
    def __init__(self, screen_width, screen_height, action_size, seed):
        """Init Agent’s models."""
        
        self.action_size = action_size
        # self.seed = random.seed(seed)
        
        # Multimodal DQN
        self.visual_model = VisualModel(seed=seed)
        self.text_model = TextModel(pretrained_embeddings, hidden_dim=64, seed=seed)

        self.dqn_net = MultimodalDQN(self.visual_model, self.text_model, action_size, seed).to(device)
    
    def visual_preprocess(self, visual_state, screen_width, screen_height):
        """Preprocess input frame before passing into agent."""
        
        resized_screen = Image.fromarray(visual_state).resize((screen_width, screen_height))
        visual_state = frame2tensor(to_torchdim(resized_screen)).to(torch.float32).to(device)

        return visual_state
    
    def text_preprocess(self, instruction):
        """Preprocess instructions before passing into agent."""
        
        text_state = instruction
        text_state = [word2vec_model.key_to_index[word] for word in text_state.split()]
        text_state = torch.tensor(text_state).long()
        text_state = text_state.unsqueeze(0)
        
        return text_state
    
    def randomize_agent(self, controller):

        positions = controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]

        position = random.choice(positions)
        controller.step(
            action="Teleport",
            position=position,
            rotation=dict(x=0, y=270, z=0),
            horizon=0,
            standing=True
        )
        
    def watch(self, controller, agent_name, floor_id, action_space):
        """Watch trained agent."""
        
        best_score = -np.inf
        
        data = [("find switch", "LightSwitch_887b121a"), 
                ("locate switch", "LightSwitch_887b121a"), 
                ("spot switch", "LightSwitch_887b121a"), 
                ("discover switch", "LightSwitch_887b121a"), 
                ("notice switch", "LightSwitch_887b121a"), 
                
                # UNCOMMON
                ("notice anything", "LightSwitch_887b121a"),
                ("leave switch", "LightSwitch_887b121a"),
                ("hit switch", "LightSwitch_887b121a"),
                
                ("find bowl", "Bowl_89852f2b"), 
                ("locate bowl", "Bowl_89852f2b"), 
                ("spot bowl", "Bowl_89852f2b"), 
                ("discover bowl", "Bowl_89852f2b"), 
                ("notice bowl", "Bowl_89852f2b"), 
                
                ("find garbage", "GarbageCan_d6916cf5"), 
                ("locate garbage", "GarbageCan_d6916cf5"), 
                ("spot garbage", "GarbageCan_d6916cf5"), 
                ("discover garbage", "GarbageCan_d6916cf5"), 
                ("notice garbage", "GarbageCan_d6916cf5"), 
               
                ("find vase", "Vase_3f629a7f"), 
                ("locate vase", "Vase_3f629a7f"), 
                ("spot vase", "Vase_3f629a7f"), 
                ("discover vase", "Vase_3f629a7f"), 
                ("notice vase", "Vase_3f629a7f"), 
               
                ("find window", "Window_dcc7eda3"), 
                ("locate window", "Window_dcc7eda3"), 
                ("spot window", "Window_dcc7eda3"), 
                ("discover window", "Window_dcc7eda3"), 
                ("notice window", "Window_dcc7eda3")]
        
        object_dict = dict(data)

        print("------")
        print(f"AGENT: {agent_name}")
        print(f"FLOOR ID: {floor_id}")
        print("------")

        # initialize the environment and state
        controller.reset(random=True)
        self.randomize_agent(controller)

        while True:

            prompt = input("\nPlease tell me the objective (switch/bowl/garbage/vase/window): ")
            
            visual_state = agent.visual_preprocess(controller.last_event.frame, 
                                               screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT)
            
            try:
                instruction, AGENT_TARGET = prompt, object_dict[prompt]
                text_state = agent.text_preprocess(instruction)
            except:
                print("Wrong instruction format.")
                continue
            
            total_score = 0

            self.dqn_net.eval()
                
            for time_step in range(1, 50):
                
                # clear_output(wait=True)
                
                # select an action using the trained dqn network
                with torch.no_grad():
                    action = self.dqn_net(visual_state, text_state).max(1)[1].view(1, 1)

                # print(f"Time Step: {time_step}, Action: {action_space[action.item()]}")
                event = controller.step(action = action_space[action.item()])
                
                _, reward, done, _ = encode_feedback(event, controller, target_name=AGENT_TARGET)

                time.sleep(0.5)

                # observe a new state
                if not done:
                    screen = controller.last_event.frame
                    resized_screen = Image.fromarray(screen).resize((SCREEN_WIDTH, SCREEN_HEIGHT))

                    next_state = frame2tensor(to_torchdim(resized_screen)).to(torch.float32).to(device)
                else:
                    next_state = None

                visual_state = next_state
                total_score += reward
                if done:
                    break
                    
            event = controller.step(action = action_space[action.item()])

            if total_score > best_score: 
                best_score = total_score

            print(f'\rInstruction: {prompt}, Total Step: {time_step}, Total Reward: {total_score}') 

agent = DQNAgent(screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT, action_size=len(action_space), seed=90)

# ## Check The Result!

# load the weights of smart agent
print("(3/5) Load Agent...")
agent.dqn_net.load_state_dict(torch.load(f'{arg1}'));

parts = arg1.split('/')
agent_name = parts[-1].split('.')[0]
floor_id = arg2

print("(4/5) Start navigating...")
agent.watch(controller, agent_name, floor_id, action_space)

print("(5/5) Terminate agent...")
controller.stop()
