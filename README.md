# Flappy Bird with Q-Learning AI

This project implements a clone of the popular Flappy Bird game with an AI that uses Q-learning to automatically learn how to play the game.

## Overview

The project consists of two main components:
- A Flappy Bird game implementation (`main.py`)
- A Q-learning AI implementation (`flappy_ai.py`)

## Game Features

- Classic Flappy Bird gameplay
- Customizable difficulty via constants
- Fallback graphics when image files are not available
- Sound effects

## AI Implementation

The AI uses Q-learning, a reinforcement learning algorithm, to learn how to play the game. The implementation:

- Discretizes the continuous game state into buckets
- Uses an epsilon-greedy strategy for exploration vs. exploitation
- Learns through trial and error by updating Q-values
- Saves and loads the Q-table between sessions

## How Q-Learning Works

1. The game state is discretized into:
   - Bird height (10 bins)
   - Bird velocity (6 bins)
   - Pipe height (10 bins)
   - Distance to next pipe (8 bins)

2. The Q-learning agent:
   - Chooses actions based on learned Q-values
   - Receives rewards for passing pipes and staying alive
   - Receives penalties for collisions
   - Updates Q-values based on the reward and next state

3. Over time, the agent learns which actions lead to better outcomes in each state.

## Training Process

Training the AI typically takes:
- 15-30 minutes for basic competency (score 5-10)
- 40-60 minutes for good performance (score 20+)
- 2+ hours for expert level (score 50+)

## Controls

### Main Game (`main.py`)
- **SPACE**: Make the bird jump
- **R**: Restart the game after game over
- **ESC/Q**: Quit the game

### AI Mode (`flappy_ai.py`)
- **SPACE**: Toggle simulation speed (60/120/240 FPS)
- **T**: Toggle between training and evaluation modes
- **S**: Manually save the Q-table
- **A**: Toggle accelerated mode (faster training)
- **ESC/Q**: Quit the game and save progress

## Running the Game

### Manual Play Mode
```
python main.py
```

### AI Training Mode
```
python flappy_ai.py
```

## Dependencies

- Python 3.x
- Pygame
- NumPy

## Tips for Faster Training

1. Run in accelerated mode (press A)
2. Increase the simulation speed (press SPACE)
3. Let it train while you do something else - the AI will save progress automatically

## Project Structure

- `main.py`: Core game implementation
- `flappy_ai.py`: Q-learning AI implementation
- `images/`: Game graphics
- `sounds/`: Game sound effects
- `q_table.pkl`: Saved AI progress (created during training)

## Future Improvements

- Deep Q-learning implementation
- More sophisticated reward functions
- Visual representation of the Q-learning process
