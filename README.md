# Flappy Bird

A Python implementation of the classic Flappy Bird game using Pygame.

## Installation

1. Make sure you have Python 3.6+ installed
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## How to Play

Run the game:
```
python main.py
```

Or:
```
./main.py
```

### Controls
- Press **SPACE** to start the game and make the bird jump
- Press **ESC** or **Q** to quit the game

## Adding Custom Assets

### Images
Place your custom image files in the `images` directory:
- `background.png` - Game background
- `bird.png` - The bird sprite
- `pipe.png` - The pipe obstacle
- `ground.png` - Ground texture

### Sounds
Place your custom sound files in the `sounds` directory:
- `wing.wav` - Bird jump sound
- `point.wav` - Point scoring sound
- `hit.wav` - Collision sound
- `die.wav` - Game over sound

If no custom assets are found, the game will use colored placeholder graphics.