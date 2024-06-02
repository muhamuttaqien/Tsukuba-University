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
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    
    plt.title(f'Training Episode: {i_episode} (Instruction: {instruction.capitalize()}, Reward: {total_score:.2f}, Time Step: {time_step})')
    plt.xlabel('Episode')
    plt.ylabel('Cummulative Rewards')
    plt.plot(scores_t.numpy(), color='green')
    
    # take 100 episode averages and plot them
    window_size = 100
    
    if len(scores_t) >= window_size:
        means = scores_t.unfold(0, window_size, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(window_size-1), means))
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
def plot_durations_w_shaded(score, i_episode, num_episodes, instruction, total_score, time_step):
    
    episode_scores.append(score)
    
    plt.figure(2)
    plt.clf()
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    
    plt.title(f'Training Episode: {i_episode} (Instruction: {instruction.capitalize()}, Reward: {total_score:.2f}, Time Step: {time_step})')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Rewards')
    
    # take 100 episode averages and plot them
    window_size = 100
    
    if len(scores_t) >= window_size:
        # calculate fixed-size window averages (means) and standard deviations
        means = scores_t.unfold(0, window_size, 1).mean(1).view(-1)
        std_deviation = scores_t.unfold(0, window_size, 1).std(1).view(-1)
        
        # pad the start with NaNs to align the plots correctly
        means = torch.cat((torch.zeros(window_size-1), means))
        std_deviation = torch.cat((torch.zeros(window_size-1), std_deviation))

        # calculate the x-axis indices for plotting
        x_values = torch.arange(len(scores_t))
        
        # plot means with standard deviation as shaded area
        plt.plot(x_values.numpy(), means.numpy(), color='red', label='Average Reward')
        plt.fill_between(x_values.numpy(), (means-std_deviation).numpy(), (means+std_deviation).numpy(), color='red', alpha=0.2, label='Standard Deviation')
    else:
        means = scores_t.unfold(0, 1, 1).mean(1).view(-1)
        plt.plot(means.numpy(), color='red', label='Average Reward')
    
    # place the legend
    plt.legend()
    
    # pause a bit so that plots are updated
    if not os.path.exists('./images/'): os.makedirs('./images/')
    plt.savefig('./images/plot_of_training_result_AI2THOR_RL.png')
    plt.pause(0.001)
    
    if is_ipython and i_episode is not num_episodes:
        display.clear_output(wait=True)
        plt.show()
    else: 
        pass
        
    return episode_scores
    
# a helper for plotting the durations of episodes
def plot_durations_wo_line(score, i_episode, num_episodes, instruction, total_score, time_step):
    
    episode_scores.append(score)
                
    plt.figure(2)
    plt.clf()
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    
    plt.title(f'Training Episode: {i_episode} (Instruction: {instruction.capitalize()}, Reward: {total_score:.2f}, Time Step: {time_step})')
    plt.xlabel('Episode')
    plt.ylabel('Cummulative Rewards')
    
    # Plot points without a line
    plt.plot(scores_t.numpy(), 'go')  # 'go' stands for green points
    
    # take 100 episode averages and plot them
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
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

def encode_feedback_find(event, controller, target_name='Bowl_89852f2b'):
    
    agent_position = event.metadata['agent']['position']
    data = controller.last_event.metadata['objects']
    
    index_location = next((index for index, item in enumerate(data) if item['name'] == target_name), None)
    is_found = data[index_location]['visible']
    
    if is_found:
        reward = +20.0
        done = True
    else:
        reward = -0.05
        done = False
           
    return None, reward, done, None

def encode_feedback_find_for_pickup(event, controller, target_name='Bowl_89852f2b', is_first=False):
    
    agent_position = event.metadata['agent']['position']
    data = controller.last_event.metadata['objects']
    
    index_location = next((index for index, item in enumerate(data) if item['name'] == target_name), None)
    is_found = data[index_location]['visible']
    
    if is_found and is_first == False:
        reward = +10.0; is_first = True
        done = False
    else:
        reward = -0.05
        done = False
           
    return None, reward, done, None, is_first

def encode_feedback_pickup(event, controller, target_name='Bowl_89852f2b'):
    
    target_data = next((obj for obj in event.metadata['objects'] if obj['name'] == target_name), None)
    
    if event.metadata['lastActionSuccess'] == True:
        if target_data['isPickedUp'] == True:
            reward = +20.0
            done = True
        else: 
            reward = -10.0
            done = False
    else: 
        reward = -0.1
        done = False
           
    return None, reward, done, None

def encode_feedback_pickup_for_go(event, controller, target_name='Bowl_89852f2b', is_first=False):
    
    target_data = next((obj for obj in event.metadata['objects'] if obj['name'] == target_name), None)
    
    if event.metadata['lastActionSuccess'] == True and is_first == False:
        if target_data['isPickedUp'] == True:
            reward = +10.0; is_first = True
            done = False
        else: 
            reward = -10.0
            done = False
    else: 
        reward = -0.1
        done = False
           
    return None, reward, done, None, is_first

def encode_feedback_go(event, controller, place_name='GarbageCan_d6916cf5'):
    
    agent_position = event.metadata['agent']['position']
    data = controller.last_event.metadata['objects']
    
    index_location = next((index for index, item in enumerate(data) if item['name'] == place_name), None)
    is_found = data[index_location]['visible']
    
    if is_found:
        reward = +20.0
        done = True
    else:
        reward = -0.05
        done = False
           
    return None, reward, done, None
    
def encode_feedback_go_for_throw(event, controller, place_name='GarbageCan_d6916cf5', is_first=False):
    
    agent_position = event.metadata['agent']['position']
    data = controller.last_event.metadata['objects']
    
    index_location = next((index for index, item in enumerate(data) if item['name'] == place_name), None)
    is_found = data[index_location]['visible']
    
    if is_found and is_first == False:
        reward = +10.0; is_first = True
        done = False
    else:
        reward = -0.05
        done = False
           
    return None, reward, done, None, is_first

    
def encode_feedback_throw(event, controller, target_name='Bowl_89852f2b', place_name='GarbageCan_d6916cf5'):
    
    target_data = next((obj for obj in event.metadata['objects'] if obj['name'] == target_name), None)
    place_data = next((obj for obj in event.metadata['objects'] if obj['name'] == place_name), None)

    if event.metadata['lastActionSuccess'] == True:
        if target_data['objectId'] in place_data['receptacleObjectIds']:
            reward = +20.0
            done = True
        else:
            reward = -10.0
            done = True
    else: 
        reward = -1
        done = False
           
    return None, reward, done, None

def encode_feedback_go_for_open(event, controller, place_name='GarbageCan_d6916cf5', is_first=False):
    
    agent_position = event.metadata['agent']['position']
    data = controller.last_event.metadata['objects']
    
    index_location = next((index for index, item in enumerate(data) if item['name'] == place_name), None)
    is_found = data[index_location]['visible']
    
    if is_found and is_first == False:
        reward = +10.0; is_first = True
        done = False
    else:
        reward = -0.05
        done = False
           
    return None, reward, done, None, is_first

def encode_feedback_open(event, controller, target_name='Bowl_89852f2b', place_name='GarbageCan_d6916cf5'):
    
    target_data = next((obj for obj in event.metadata['objects'] if obj['name'] == target_name), None)
    place_data = next((obj for obj in event.metadata['objects'] if obj['name'] == place_name), None)

    if event.metadata['lastActionSuccess'] == True:
        if place_data['isOpen']:
            reward = +20.0
            done = True
        else:
            reward = -10.0
            done = True
    else: 
        reward = -1
        done = False
           
    return None, reward, done, None

def encode_feedback_open_for_put(event, controller, target_name='Bowl_89852f2b', place_name='GarbageCan_d6916cf5', is_first=False):
    
    target_data = next((obj for obj in event.metadata['objects'] if obj['name'] == target_name), None)
    place_data = next((obj for obj in event.metadata['objects'] if obj['name'] == place_name), None)

    if event.metadata['lastActionSuccess'] == True:
        if place_data['isOpen'] and is_first == False:
            reward = +10.0; is_first = True
            done = False
        else:
            reward = -5.0
            done = False
    else: 
        reward = -1
        done = False
           
    return None, reward, done, None, is_first

def encode_feedback_put(event, controller, target_name='Bowl_89852f2b', place_name='GarbageCan_d6916cf5'):
    
    target_data = next((obj for obj in event.metadata['objects'] if obj['name'] == target_name), None)
    place_data = next((obj for obj in event.metadata['objects'] if obj['name'] == place_name), None)

    if event.metadata['lastActionSuccess'] == True:
        if target_data['objectId'] in place_data['receptacleObjectIds']:
            reward = +20.0
            done = True
        else:
            reward = -10.0
            done = True
    else: 
        reward = -1
        done = False
           
    return None, reward, done, None
