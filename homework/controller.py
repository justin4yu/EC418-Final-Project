import pystk
import numpy as np
import random
import json


# ---------------------------- Q Learning ------------------------------
Q = {}

# Hyperparameters
alpha = 0.1  
gamma = 0.9  
epsilon = 0.9  
epsilon_decay = 0.995
epsilon_min = 0.05

actions = ['steer_left', 'steer_right', 'brake', 'accelerate']


def get_reward(aim_point, current_vel, target_vel=10, collision=False):
    """
    Define a reward function for Q-learning.
    """
    progress_reward = -abs(aim_point[0])  
    velocity_reward = -abs(current_vel - target_vel)  
    collision_penalty = -10 if collision else 0  
    return progress_reward + velocity_reward + collision_penalty


def initialize_q(state):
    """
    Initialize Q-values for a new state.
    """
    if state not in Q:
        Q[state] = {action: 0.0 for action in actions}


def choose_action(state):
    """
    Choose an action using an epsilon-greedy policy.
    """
    initialize_q(state) 

    if np.random.rand() < epsilon:
        action = random.choice(actions)
    else:
        action = max(Q[state], key=Q[state].get)  

    print(f"State: {state}, Action: {action}, Q-Values: {Q[state]}")
    return action


def update_q_value(state, action, reward, next_state):

    initialize_q(next_state)  

    max_next_q = max(Q[next_state].values())
    old_q_value = Q[state][action]
    Q[state][action] += alpha * (reward + gamma * max_next_q - old_q_value) 

    print(f"Updated Q-Value: State: {state}, Action: {action}, Old Q-Value: {old_q_value}, New Q-Value: {Q[state][action]}")


def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=10):
    """
    Q-learning control logic.
    """
    action = pystk.Action()

    state = (round(aim_point[0], 1), round(current_vel, 0))

    chosen_action = choose_action(state)

    if current_vel < target_vel:
        action.acceleration = 1.0
    else:
        action.acceleration = 0.0

    if chosen_action == 'steer_left':
        action.steer = np.clip(-steer_gain * aim_point[0], -1, 1)
        action.acceleration = 1.0
    elif chosen_action == 'steer_right':
        action.steer = np.clip(steer_gain * aim_point[0], -1, 1)
        action.acceleration = 1.0
    elif chosen_action == 'accelerate':
        action.steer = np.clip(steer_gain * aim_point[0], -1, 1)
        action.acceleration = 1.0 if current_vel < target_vel else 0.0
        action.brake = False
    elif chosen_action == 'brake':
        action.acceleration = -1.0

    action.drift = abs(aim_point[0]) > skid_thresh
    action.nitro = abs(aim_point[0]) < 0.5

    collision = abs(aim_point[0]) > 1.0 

    reward = get_reward(aim_point, current_vel, target_vel, collision)

    next_aim_point = (aim_point[0] - action.steer * 0.1, aim_point[1])
    next_velocity = max(current_vel + (action.acceleration if not action.brake else -1.0), 0)
    next_state = (round(next_aim_point[0], 1), round(next_velocity, 0))

    update_q_value(state, chosen_action, reward, next_state)

    global epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    return action



# ---------------------------- TD Learning w Linear Approx ------------------------------
# import numpy as np
# import random

# # Hyperparameters
# ALPHA = 0.01  # Lowered learning rate
# GAMMA = 0.9
# EPSILON = 0.25

# # Initialize weights for each action
# theta = {
#     'steer_left': np.zeros(3),
#     'steer_right': np.zeros(3),
#     'accelerate': np.zeros(3),
#     'brake': np.zeros(3)
# }

# # Feature extraction function
# def extract_features(aim_point, current_vel):
#     # Normalize features to avoid large values in the weight updates
#     return np.array([1, aim_point[0], current_vel / 100])

# # Get the value (Q-value) for a given state-action pair
# def get_value(features, action):
#     return np.dot(theta[action], features)

# # Choose an action based on epsilon-greedy policy
# def choose_action(state_features):
#     if random.random() < EPSILON:
#         return random.choice(['steer_left', 'steer_right', 'accelerate', 'brake'])
#     else:
#         q_values = {action: get_value(state_features, action) for action in theta}
#         return max(q_values, key=q_values.get)

# # Update the weights (theta) using the TD error
# def update_theta(state_features, reward, next_state_features, chosen_action):
#     global theta

#     next_q_values = {action: get_value(next_state_features, action) for action in theta}
#     target = reward + GAMMA * max(next_q_values.values())

#     td_error = target - get_value(state_features, chosen_action)

#     theta[chosen_action] += ALPHA * td_error * state_features

#     print(f"Updated Theta for {chosen_action}: {theta[chosen_action]}")

# def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.1, target_vel=40):
#     action = pystk.Action()
    
#     state_features = extract_features(aim_point, current_vel)
    
#     chosen_action = choose_action(state_features)
#     print(f"State Features: {state_features}, Chosen Action: {chosen_action}")
    
#     if current_vel < 1.0:
#         action.acceleration = 5.0  # Apply consistent forward motion to start

#     # Apply chosen action
#     if chosen_action == 'steer_left':
#         action.steer = np.clip(steer_gain * aim_point[0], -1, 1)  # Steer left
#         action.acceleration = 3.0
#     elif chosen_action == 'steer_right':
#         action.steer = np.clip(steer_gain * aim_point[0], -1, 1)  # Steer right
#         action.acceleration = 3.0
#     elif chosen_action == 'accelerate':
#         action.acceleration = 5.0  # Accelerate 
#         action.brake = 0
#     elif chosen_action == 'brake':
#         action.brake = True  # Brake
#         action.acceleration = 0

#     # Enable drift for sharp turns
#     action.drift = abs(aim_point[0]) > skid_thresh

#     reward = -abs(aim_point[0]) - abs(current_vel - target_vel)

#     next_aim_point = (aim_point[0] - action.steer, aim_point[1])
#     next_state_features = extract_features(next_aim_point, current_vel)
    
#     update_theta(state_features, reward, next_state_features, chosen_action)
    
#     return action





# import pystk
# import numpy as np
# import random

# # Hyperparameters
# ALPHA = 0.01  # Learning rate
# GAMMA = 0.9   # Discount factor
# EPSILON = 0.95 # Exploration rate
# EPSILON_DECAY = 0.995
# EPSILON_MIN = 0.01

# # Initialize weights for actions (global for reuse across runs)
# theta = {
#     'steering': np.zeros(3),
#     'accelerate': np.zeros(3),
#     'brake': np.zeros(3)
# }

# prev_action = pystk.Action()  # For smoothness reward


# # Feature extraction
# def extract_features(aim_point, current_vel):
#     """
#     Extract features for linear approximation.
#     :param aim_point: Tuple representing aim point.
#     :param current_vel: Current velocity of the kart.
#     :return: Feature vector as a numpy array.
#     """
#     return np.array([1, aim_point[0], current_vel / 100])  # Include bias, aim_point, normalized velocity


# # Calculate Q-value for a given action
# def get_value(features, action):
#     return np.dot(theta[action], features)


# # Choose an action using epsilon-greedy
# def choose_action(state_features):
#     global EPSILON
#     if random.random() < EPSILON: 
#         return random.choice(list(theta.keys()))
#     else:  
#         q_values = {action: get_value(state_features, action) for action in theta}
#         return max(q_values, key=q_values.get)


# # Update theta using TD(0)
# def update_theta(state_features, reward, next_state_features, chosen_action):
#     global theta

#     next_q_values = {action: get_value(next_state_features, action) for action in theta}
#     target = reward + GAMMA * max(next_q_values.values())
#     td_error = target - get_value(state_features, chosen_action)

#     theta[chosen_action] += ALPHA * td_error * state_features

#     print(f"Updated Theta for {chosen_action}: {theta[chosen_action]}")


# # Reward function
# def get_reward(aim_point, current_vel, target_vel=20, acceleration=0, distance_down_track=0, track_length=1):
#     """
#     Calculate the reward for the current state.
#     :param aim_point: The aiming point of the kart on the track.
#     :param current_vel: The current velocity of the kart.
#     :param target_vel: The target velocity.
#     :param acceleration: The acceleration action taken.
#     :param distance_down_track: Distance traveled along the track.
#     :param track_length: Total length of the track.
#     :return: The reward value.
#     """
#     progress_reward = (distance_down_track / track_length) * 10 
#     velocity_reward = -abs(current_vel - target_vel)  
#     smoothness_reward = -abs(prev_action.steer)  
#     drift_penalty = -abs(aim_point[0]) if abs(aim_point[0]) > 0.2 else 0 
#     return progress_reward + velocity_reward + smoothness_reward + drift_penalty


# # Control function
# def control(aim_point, current_vel, steer_gain=5, skid_thresh=0.3, target_vel=20, distance_down_track=0, track_length=1):
#     global prev_action, EPSILON
#     action = pystk.Action()

#     state_features = extract_features(aim_point, current_vel)
#     chosen_action = choose_action(state_features)

#     if current_vel < target_vel:
#         action.acceleration = 5.0
#     else:
#         action.acceleration = 0.0

#     if chosen_action == 'steering':
#         action.steer = np.clip(steer_gain * aim_point[0], -1, 1)
#         action.acceleration = 1.0
#         action.nitro = abs(aim_point[0]) < 0.2
#     elif chosen_action == 'accelerate':
#         action.steer = np.clip((steer_gain*2) * aim_point[0], -1, 1)
#         action.acceleration = 3.0
#         action.brake = False
#     elif chosen_action == 'brake':
#         action.acceleration = -1.0

#     action.drift = abs(aim_point[0]) > skid_thresh

#     reward = get_reward(aim_point, current_vel, target_vel, distance_down_track=distance_down_track, track_length=track_length)

#     next_velocity = current_vel + action.acceleration
#     next_aim_point = (aim_point[0] - action.steer, aim_point[1])
#     next_state_features = extract_features(next_aim_point, next_velocity)

#     update_theta(state_features, reward, next_state_features, chosen_action)

#     # Decay epsilon for less exploration over time
#     EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

#     # Save the current action for smoothness calculation
#     prev_action = action

#     return action







# ---------------------------- TD Learning w Linear Approx Double Action ------------------------------
# import numpy as np

# # Hyperparameters
# ALPHA = 0.1
# GAMMA = 0.9
# EPSILON = 0.95

# # Initialize theta for each action, each having its own weight vector
# theta = {
#     'steer_left': np.random.randn(3, 1) * 0.01,
#     'steer_right': np.random.randn(3, 1) * 0.01,
#     'accelerate': np.random.randn(3, 1) * 0.01,
#     'brake': np.random.randn(3, 1) * 0.01
# }

# # Feature extraction function returning a column vector (scale features if necessary)
# def extract_features(aim_point, current_vel):
#     return np.array([[1], [aim_point[0] / 10], [current_vel / 1000]])  # Normalize values

# # Get value from feature vector and theta (column vectors)
# def get_value(features, action):
#     return np.dot(theta[action].T, features)  # Transpose theta to ensure it's a 1x3 vector

# def choose_action(state_features):
#     """Choose the top two actions based on the value of the actions."""
#     q_values = {}
    
#     # Calculate Q-values for each action
#     for action in theta:
#         q_values[action] = get_value(state_features, action)
    
#     print(f'Q-Values: {q_values}')
    
#     # Find the top two actions
#     sorted_actions = []
#     while len(sorted_actions) < 2 and len(q_values) > 0:
#         # Find the action with the highest Q-value
#         max_action = None
#         max_value = float('-inf')
#         for action, value in q_values.items():
#             if value > max_value:
#                 max_value = value
#                 max_action = action
#         # Add it to the sorted list and remove from q_values
#         if max_action is not None:
#             sorted_actions.append(max_action)
#             del q_values[max_action]
    
#     return sorted_actions

# # Update theta using the TD error
# def update_theta(state_features, reward, next_state_features, chosen_action, next_action):
#     global theta
#     current_value = get_value(state_features, chosen_action)
#     next_value = get_value(next_state_features, next_action)
    
#     # Calculate TD error
#     td_error = reward + GAMMA * next_value - current_value
    
#     # Update theta for the chosen action only
#     theta[chosen_action] += ALPHA * td_error * state_features
    
#     print(f"Updated Theta for {chosen_action}: {theta[chosen_action].flatten()}")

# def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
#     """Control function with integrated learning and debugging for multiple actions."""
#     action = pystk.Action()
    
#     # Extract features for the current state
#     state_features = extract_features(aim_point, current_vel)
    
#     # Choose top two actions
#     chosen_actions = choose_action(state_features)
#     print(f"Current State Features: {state_features.T}, Chosen Actions: {chosen_actions}")
    
#     # Apply the combined actions
#     if 'steer_left' in chosen_actions:
#         action.steer = np.clip(-steer_gain * aim_point[0], -1, 1)
#     if 'steer_right' in chosen_actions:
#         action.steer = np.clip(steer_gain * aim_point[0], -1, 1)
#     if 'accelerate' in chosen_actions:
#         action.acceleration = 1.0 if current_vel < target_vel else 0.0
#     if 'brake' in chosen_actions:
#         action.brake = current_vel > target_vel
    
#     # Compute reward
#     reward = -abs(aim_point[0]) - abs(current_vel - target_vel)
    
#     # Approximate next state
#     next_aim_point = (aim_point[0] - action.steer, aim_point[1])
#     next_state_features = extract_features(next_aim_point, current_vel)
#     print(f"Next State Features: {next_state_features.T}")
    
#     # Choose next actions
#     next_actions = choose_action(next_state_features)
    
#     # Update the agent's weights for both chosen actions
#     for chosen_action in chosen_actions:
#         update_theta(state_features, reward, next_state_features, chosen_action, next_actions[0])  # Use the highest next_action
    
#     return action







if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=3000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)

# import os
# print('current directory')
# print(os.getcwd())



# import pickle
# import pystk
# import numpy as np

# class PIDController:
#     def __init__(self, Kp, Ki, Kd):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.Kd = Kd
#         self.integral = 0
#         self.previous_error = 0

#     def calculate(self, error, dt):
#         self.integral += error * dt
#         derivative = (error - self.previous_error) / dt
#         self.previous_error = error
#         return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

#     def reset_integral(self):
#         self.integral = 0

#     def save_state(self, filepath):
#         with open(filepath, 'wb') as file:
#             pickle.dump({'integral': self.integral, 'previous_error': self.previous_error}, file)

#     def load_state(self, filepath):
#         try:
#             with open(filepath, 'rb') as file:
#                 state = pickle.load(file)
#                 self.integral = state['integral']
#                 self.previous_error = state['previous_error']
#         except FileNotFoundError:
#             pass


# steering_pid = PIDController(Kp=3, Ki=0.1, Kd=1.5)
# speed_pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.2)

# steering_pid.load_state('steering_pid.pkl')
# speed_pid.load_state('speed_pid.pkl')


# def control(aim_point, current_vel, steer_gain=5, skid_thresh=0.2, target_vel=20, dt=1.0):
#     global prev_action
#     action = pystk.Action()

#     if current_vel < 1.0 and abs(aim_point[0]) > 0.7: 
#         steering_pid.reset_integral()  
#         speed_pid.reset_integral()  
#         action.rescue = True 
#         return action

#     steering_error = aim_point[0]  
#     speed_error = target_vel - current_vel  

#     action.steer = np.clip(steering_pid.calculate(steering_error, dt), -1, 1)
#     action.acceleration = np.clip(speed_pid.calculate(speed_error, dt), 0, 5)

#     action.brake = current_vel > target_vel and abs(aim_point[0]) > skid_thresh

#     action.drift = abs(aim_point[0]) > skid_thresh

#     return action


# def save_pid_states():
#     steering_pid.save_state('steering_pid.pkl')
#     speed_pid.save_state('speed_pid.pkl')


# if __name__ == '__main__':
#     from utils import PyTux
#     from argparse import ArgumentParser

#     def test_controller(args):
#         pytux = PyTux()
#         for t in args.track:
#             steps, how_far = pytux.rollout(
#                 track=t,
#                 controller=control,
#                 max_frames=3000,
#                 verbose=args.verbose
#             )
#             print(f"Steps: {steps}, Distance Covered: {how_far}")
#         pytux.close()

#         # Save PID states after testing
#         save_pid_states()



#     parser = ArgumentParser()
#     parser.add_argument('track', nargs='+')
#     parser.add_argument('-v', '--verbose', action='store_true')
#     args = parser.parse_args()
#     test_controller(args)
