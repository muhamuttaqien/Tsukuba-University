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
def plot_durations(score, i_episode, num_episodes, instruction, total_score, time_step):
    
    episode_scores.append(score)
                
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_scores, dtype=torch.float)
    
    plt.title(f'Training Episode: {i_episode} (Instruction: {instruction.capitalize()}, Reward: {total_score:.2f}, Time Step: {time_step})')
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
    else: 
        return

# a helper for plotting the durations of episodes
def plot_durations_wo_line(score, i_episode, num_episodes, instruction, total_score, time_step):
    
    episode_scores.append(score)
                
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_scores, dtype=torch.float)
    
    plt.title(f'Training Episode: {i_episode} (Instruction: {instruction.capitalize()}, Reward: {total_score:.2f}, Time Step: {time_step})')
    plt.xlabel('Episode')
    plt.ylabel('Cummulative Rewards')
    
    # Plot points without a line
    plt.plot(durations_t.numpy(), 'go')  # 'go' stands for green points
    
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
    else:
        return

def encode_feedback_nav(event, controller, target_name='Bowl_89852f2b'):
    
    data = controller.last_event.metadata['objects']
    
    index_location = next((index for index, item in enumerate(data) if item['name'] == target_name), None)
    is_found = data[index_location]['visible']
    
    if is_found:
        reward = +10.0 # -0.05 # +20.0
        done = False # True
    else:
        reward = -0.05
        done = False
           
    return None, reward, done, None

def encode_feedback_nav_for_pickup(event, controller):
    
    reward = -0.05
    done = False
           
    return None, reward, done, None

def encode_feedback_pickup(event, controller, target_name='Bowl_89852f2b'):
    
    target_data = next((obj for obj in event.metadata['objects'] if obj['name'] == target_name), None)
    
    if event.metadata['lastActionSuccess'] == True:
        if target_data['isPickedUp'] == True:
            reward = +20.0 # +20.0
            done = False
        else: 
            reward = -10.0
            done = False
    else: 
        reward = -0.1
        done = False
           
    return None, reward, done, None

def encode_feedback_drop(event, controller, target_name='Bowl_89852f2b', place_name='DiningTable_00be542e'):
    
    target_data = next((obj for obj in event.metadata['objects'] if obj['name'] == target_name), None)
    place_data = next((obj for obj in event.metadata['objects'] if obj['name'] == place_name), None)
    
    if event.metadata['lastActionSuccess'] == True:
        if target_data['objectId'] in place_data['receptacleObjectIds']:
            reward = +40.0 # +40.0
            done = True
        else:
            reward = -20.0
            done = True
    else: 
        reward = -0.1
        done = False
           
    return None, reward, done, None

def encode_feedback_terminate(event, controller, instruction, target_name='Bowl_89852f2b', place_name='DiningTable_00be542e'):
    
    target_data = next((obj for obj in event.metadata['objects'] if obj['name'] == target_name), None)
    place_data = next((obj for obj in event.metadata['objects'] if obj['name'] == place_name), None)
    
    if instruction == "find bowl take it then place it on the table":
        if event.metadata['lastActionSuccess'] == True: # terminate when the bowl is on the table
            if target_data['objectId'] in place_data['receptacleObjectIds']:
                reward = +30.0
                done = True
            else: # terminate not in the right time
                reward = -0.5
                done = False  
        else: # terminate not in the right time
            reward = -0.5
            done = False
        
    elif instruction == "find bowl take it":
        if event.metadata['lastActionSuccess'] == True: # terminate when the bowl is taken
            if target_data['isPickedUp'] == True:
                reward = +70.0
                done = True
            else: 
                reward = -0.5
                done = False
        else: # terminate not in the right time
            reward = -0.5
            done = False
            
    elif instruction == "find bowl":
        data = controller.last_event.metadata['objects']
    
        index_location = next((index for index, item in enumerate(data) if item['name'] == target_name), None)
        is_found = data[index_location]['visible']
        
        if is_found:
            reward = +90.0
            done = True
        else: 
            reward = -0.5
            done = False
            
    return None, reward, done, None
