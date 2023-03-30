# flappy_bird_RL
Implementation of a Q-Learning Agent and an Expected Sarsa agent for the game Flappy Bird.

To launch a game of flappy bird, you need to choose the trained agent to be imported with pickle:
 
```
with open('models/q_learning.pkl', 'rb') as pickle_file:
        Q = pickle.load(pickle_file)
```

And launch the main.py file through the CLI:

`python main.py`
