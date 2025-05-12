#!/usr/bin/env python3
import os
import sys
import random
import pygame

# Initialize pygame
pygame.init()
pygame.mixer.init()

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRAVITY = 0.5
BIRD_JUMP = -10
PIPE_SPEED = 3
PIPE_GAP = 150
PIPE_FREQUENCY = 1800  
GROUND_HEIGHT = 100
BIRD_WIDTH = 40
BIRD_HEIGHT = 30
PIPE_WIDTH = 60
FPS = 60

# Colors (for fallback graphics)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)
SKYBLUE = (135, 206, 235)
BROWN = (139, 69, 19)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Flappy Bird')
clock = pygame.time.Clock()

# Load images with fallbacks
def load_image(filename, width, height, color=None):
    filepath = os.path.join('images', filename)
    try:
        image = pygame.image.load(filepath)
        return pygame.transform.scale(image, (width, height))
    except:
        # Create a colored rectangle as fallback
        surface = pygame.Surface((width, height))
        if color:
            surface.fill(color)
        return surface

# Modified to create a proper sized pipe image
pipe_img = None
try:
    pipe_path = os.path.join('images', 'pipe.png')
    pipe_img = pygame.image.load(pipe_path)
    pipe_img = pygame.transform.scale(pipe_img, (PIPE_WIDTH, 300))  # Scale to reasonable height
except:
    # Create a colored rectangle as fallback with fixed height
    pipe_img = pygame.Surface((PIPE_WIDTH, 300))
    pipe_img.fill(GREEN)

# Load the other assets as before
background = load_image('background.png', SCREEN_WIDTH, SCREEN_HEIGHT, SKYBLUE)
bird_img = load_image('bird.png', BIRD_WIDTH, BIRD_HEIGHT, BLUE)
ground_img = load_image('ground.png', SCREEN_WIDTH, GROUND_HEIGHT, BROWN)

# Load sounds with fallbacks
def load_sound(filename):
    filepath = os.path.join('sounds', filename)
    try:
        return pygame.mixer.Sound(filepath)
    except:
        # Return a dummy sound object
        return type('DummySound', (), {'play': lambda: None})

# Load sounds
sound_wing = load_sound('wing.wav')
sound_point = load_sound('point.wav')
sound_hit = load_sound('hit.wav')
sound_die = load_sound('die.wav')

# Bird class
class Bird:
    def __init__(self):
        self.x = SCREEN_WIDTH // 4
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.image = bird_img
        self.rect = pygame.Rect(self.x, self.y, BIRD_WIDTH, BIRD_HEIGHT)
    
    def jump(self):
        self.velocity = BIRD_JUMP
        sound_wing.play()
    
    def update(self):
        # Apply gravity
        self.velocity += GRAVITY
        self.y += self.velocity
        
        # Update rectangle position
        self.rect.y = self.y
    
    def draw(self):
        screen.blit(self.image, (self.x, self.y))

# Pipe class
class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        # Create a proper gap:
        # 1. Calculate the available vertical space (excluding the ground)
        available_space = SCREEN_HEIGHT - GROUND_HEIGHT
        # 2. Gap is fixed at PIPE_GAP
        # 3. Calculate how much space is left for the top and bottom pipes combined
        pipe_space = available_space - PIPE_GAP
        # 4. Randomly decide how to distribute this space between top and bottom pipes
        self.top_height = random.randint(80, pipe_space - 80)  # At least 80px for each pipe
        self.passed = False
        
        # Create top and bottom pipe rectangles with proper spacing
        self.top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.top_height)
        self.bottom_rect = pygame.Rect(
            self.x, 
            self.top_height + PIPE_GAP, 
            PIPE_WIDTH, 
            available_space - self.top_height - PIPE_GAP
        )
    
    def update(self):
        self.x -= PIPE_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x
    
    def draw(self):
        # Draw top pipe (flipped) - Fix: only use the portion we need
        top_pipe = pygame.transform.flip(pipe_img, False, True)
        # Draw the top pipe multiple times to fill the height if needed
        remaining_height = self.top_height
        segment_height = pipe_img.get_height()
        
        while remaining_height > 0:
            draw_height = min(segment_height, remaining_height)
            # Create a subsurface for the portion we need
            segment = top_pipe.subsurface((0, 0, PIPE_WIDTH, draw_height))
            screen.blit(segment, (self.x, self.top_height - remaining_height))
            remaining_height -= draw_height
        
        # Draw bottom pipe - position it correctly based on gap
        bottom_pipe_y = self.top_height + PIPE_GAP
        remaining_height = SCREEN_HEIGHT - GROUND_HEIGHT - bottom_pipe_y
        
        # Draw the bottom pipe multiple times to fill the height if needed
        current_y = bottom_pipe_y
        while remaining_height > 0:
            draw_height = min(segment_height, remaining_height)
            # Create a subsurface for the portion we need
            segment = pipe_img.subsurface((0, 0, PIPE_WIDTH, draw_height))
            screen.blit(segment, (self.x, current_y))
            current_y += draw_height
            remaining_height -= draw_height

# Game class
class Game:
    def __init__(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.game_started = False
        self.ground_scroll = 0
        self.last_pipe = pygame.time.get_ticks() - PIPE_FREQUENCY
        self.font = pygame.font.Font(None, 50)
    
    def reset(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.game_started = False
        self.last_pipe = pygame.time.get_ticks() - PIPE_FREQUENCY
    
    def update(self):
        if not self.game_started or self.game_over:
            return
        
        # Update bird
        self.bird.update()
        
        # Ground collision
        if self.bird.y > SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_HEIGHT:
            self.bird.y = SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_HEIGHT
            self.game_over = True
            sound_hit.play()
            sound_die.play()
        
        # Ceiling collision
        if self.bird.y < 0:
            self.bird.y = 0
            self.bird.velocity = 0
        
        # Create new pipes
        time_now = pygame.time.get_ticks()
        if time_now - self.last_pipe > PIPE_FREQUENCY:
            self.pipes.append(Pipe())
            self.last_pipe = time_now
        
        # Update and check pipes
        for pipe in self.pipes:
            pipe.update()
            
            # Check for collisions
            if pipe.top_rect.colliderect(self.bird.rect) or pipe.bottom_rect.colliderect(self.bird.rect):
                self.game_over = True
                sound_hit.play()
                sound_die.play()
            
            # Check if pipe has passed the bird
            if pipe.x + PIPE_WIDTH < self.bird.x and not pipe.passed:
                pipe.passed = True
                self.score += 1
                sound_point.play()
            
            # Remove pipes that are off screen
            if pipe.x < -PIPE_WIDTH:
                self.pipes.remove(pipe)
        
        # Update ground scroll - fixed to prevent disappearing
        self.ground_scroll = (self.ground_scroll - PIPE_SPEED)
        if self.ground_scroll <= -SCREEN_WIDTH:
            self.ground_scroll = 0
    
    def draw(self):
        # Draw background
        screen.blit(background, (0, 0))
        
        # Draw pipes
        for pipe in self.pipes:
            pipe.draw()
        
        # Draw ground (scrolling)
        screen.blit(ground_img, (self.ground_scroll, SCREEN_HEIGHT - GROUND_HEIGHT))
        screen.blit(ground_img, (self.ground_scroll + SCREEN_WIDTH, SCREEN_HEIGHT - GROUND_HEIGHT))
        
        # Draw bird
        self.bird.draw()
        
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        # Draw game over or start message
        if not self.game_started:
            start_text = self.font.render("Press SPACE to start", True, WHITE)
            screen.blit(start_text, (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2))
        elif self.game_over:
            over_text = self.font.render("Game Over!", True, WHITE)
            restart_text = self.font.render("Press R to restart", True, WHITE)
            screen.blit(over_text, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 - 50))
            screen.blit(restart_text, (SCREEN_WIDTH//2 - 125, SCREEN_HEIGHT//2 + 50))

def main():
    game = Game()
    running = True
    
    while running:
        clock.tick(FPS)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                
                if event.key == pygame.K_SPACE:
                    if not game.game_over:
                        if not game.game_started:
                            game.game_started = True
                        game.bird.jump()
                
                if event.key == pygame.K_r and game.game_over:
                    game.reset()
        
        # Update game state
        game.update()
        
        # Draw everything
        game.draw()
        
        # Update display
        pygame.display.update()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
