import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
    
plt.ion()


def to_torchdim(state):
    
    state = np.transpose(state, (2, 0, 1))
    state = np.expand_dims(state, axis=0)
    
    return state

def frame2tensor(frame):
    
    frame_copy = np.copy(frame)
    tensor = torch.from_numpy(frame_copy)
    
    return tensor

episode_scores = []

# a helper for plotting the durations of episodes
def plot_durations(score, i_episode, num_episodes):
    
    episode_scores.append(score)
                
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_scores, dtype=torch.float)
    
    plt.title(f'Training Episode: {i_episode}')
    plt.xlabel('Episode')
    plt.ylabel('Cummulative Rewards')
    plt.plot(durations_t.numpy(), color='green')
    
    # take 100 episode averages and plot them
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), color='red')
    
    # pause a bit so that plots are updated
    if not os.path.exists('./images/'): os.makedirs('./images/')
    plt.savefig('./images/plot_of_training_result_AI2THOR_RL.png')
    plt.pause(0.001)
    
    if is_ipython and i_episode is not num_episodes:
        display.clear_output(wait=True)
        plt.show()
    else: return

def encode_feedback(event, controller, target_name):
    
    agent_position = event.metadata["agent"]["position"]
    data = controller.last_event.metadata["objects"]
    
    index_location = next((index for index, item in enumerate(data) if item['name'] == target_name), None)
    is_found = data[index_location]["visible"]
    
    if is_found:
        reward = +10.0
    else:
        reward = -0.05
           
    return None, reward, is_found, None

