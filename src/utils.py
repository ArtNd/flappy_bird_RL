import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def train_agent(agent_class, agent_info: dict, epochs: int):
    agent = agent_class
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)

    all_reward_sums = [] # Contains sum of rewards during episode
    all_state_visits = [] # Contains state visit counts during the last 10 episodes

    agent_info["seed"] = 0
    agent_info["policy"] = None
    agent.agent_init(agent_info)

    num_runs = epochs # The number of runs
    num_episodes = 500 # The number of episodes in each run

    # Iteration over the number of runs
    for run in range(num_runs):

        # Set the seed value to the current run index
        agent_info["seed"] = run

        # Initialize the environment
        # Returns (obs: (x_dist,y_dist), info: {"score", "player", "distance"})
        state, info = env.reset()

        # Set done to False
        done = False

        reward_sums = []
        state_visits = {}

        # Iterate over the number of episodes
        for episode in range(num_episodes):
            if episode == 0:
        
                # Keep track of the visited states
                state, info = env.reset()
                action = agent.agent_start(state)

                state_visits[state] = 1
                state, reward, done, _, info = env.step(action)
                reward_sums.append(reward)

            else:
                while not done:
                    action = agent.agent_step(reward, state)

                    if state not in state_visits: 
                        state_visits[state] = 1
                    else:
                        state_visits[state] += 1

                    state, reward, done, _, info = env.step(action)
                    reward_sums.append(reward)

                    # If terminal state
                    if done:
                        action = agent.agent_end(reward, state)
                        break

        all_reward_sums.append(np.sum(reward_sums))
        all_state_visits.append(state_visits)
        
    return agent.q, all_reward_sums

def plot_policy(q_values, title="Flappy Bird Policy"):
    """
    Plot the policy for the Flappy Bird environment.

    Args:
        q_values (dict): Dictionary of state-action values.
        title (str): Title of the plot.

    Returns:
        None
    """
    def get_Z(x, y):
        if (x, y) in q_values:
            # Find key_value pair with highest Q value in the dictionary
            pi = max(q_values[(x, y)], key=q_values[(x, y)].get)
            return pi
        else:
            return 2

    def get_figure(ax):
        x_range = np.arange(14, 0, -1)
        y_range = np.arange(-11, 12, 1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x, y) for x in x_range] for y in y_range])
        cmap = plt.get_cmap('Set2', 3)
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        surf = ax.imshow(Z, cmap=cmap, norm=norm, extent=[0.5, 13.5, -10.5, 10.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        ax.set_xlabel("x_dist")
        ax.set_ylabel("y_dist")
        ax.grid(color="w", linestyle="-", linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, cmap=cmap, norm=norm, ticks=[0, 1, 2], cax=cax)
        cbar.ax.set_yticklabels(["0 (Idle)", "1 (Flap)", "2 (Unexplored)"])

    fig = plt.figure()
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    get_figure(ax)
    plt.show()

def plot_state_values(V):
    def get_Z(x, y):
        if (x,y) in V:
            return V[x,y]
        else:
            return 0

    def get_figure(ax):
        x_range = np.arange(0, 14)
        y_range = np.arange(-11, 11)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('State-Value Graph')
    get_figure(ax)
    plt.show()

def get_policy(q_table):
    policy = {}
    for k, v in q_table.items():
        pi = max(q_table[k], key=q_table[k].get)
        policy[k] = pi
    return policy

def get_state_value(q_table):
    policy = {}
    for k, v in q_table.items():
        pi = max(q_table[k], key=q_table[k].get)
        policy[k] = q_table[k][pi]
    return policy