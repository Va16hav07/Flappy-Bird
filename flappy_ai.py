import numpy as np
import random
import pygame
import os
import sys
import pickle
# Import all necessary constants from main.py
from main import (
    Bird, Pipe, Game, SCREEN_WIDTH, SCREEN_HEIGHT, GROUND_HEIGHT, FPS, 
    PIPE_WIDTH, WHITE, screen, PIPE_FREQUENCY, PIPE_SPEED, PIPE_GAP,
    background, ground_img, bird_img
)

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Q-Learning agent for Flappy Bird
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        """
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Q-table: state -> {0: q_value, 1: q_value}
        
        # State space discretization
        self.height_bins = 10  # Bird height buckets
        self.velocity_bins = 6  # Bird velocity buckets
        self.pipe_height_bins = 10  # Pipe height buckets
        self.distance_bins = 8  # Distance to pipe buckets
        
        # Stats
        self.training_iterations = 0
        
        # Training metrics
        self.training_start_time = pygame.time.get_ticks()
        self.last_progress_report = 0
        self.report_interval = 60000  # Report progress every minute
    
    def discretize_state(self, bird_y, bird_vel, pipe_height, pipe_distance):
        """
        Convert continuous state to discrete buckets
        Returns tuple that can be used as dictionary key
        """
        # Normalize and discretize bird height (y-position)
        bird_y_norm = bird_y / SCREEN_HEIGHT
        bird_y_bin = min(self.height_bins - 1, int(bird_y_norm * self.height_bins))
        
        # Normalize and discretize bird velocity
        # Bird velocity is typically between -10 (upward) and +10 (downward)
        bird_vel_norm = (bird_vel + 10) / 20  # Map from [-10, 10] to [0, 1]
        bird_vel_norm = max(0, min(1, bird_vel_norm))  # Clip to [0, 1]
        bird_vel_bin = min(self.velocity_bins - 1, int(bird_vel_norm * self.velocity_bins))
        
        # Normalize and discretize pipe height (top pipe)
        pipe_height_norm = pipe_height / SCREEN_HEIGHT
        pipe_height_bin = min(self.pipe_height_bins - 1, int(pipe_height_norm * self.pipe_height_bins))
        
        # Normalize and discretize distance to pipe
        max_distance = SCREEN_WIDTH  # Maximum possible distance
        pipe_distance_norm = pipe_distance / max_distance
        pipe_distance_bin = min(self.distance_bins - 1, int(pipe_distance_norm * self.distance_bins))
        
        return (bird_y_bin, bird_vel_bin, pipe_height_bin, pipe_distance_bin)
    
    def get_action(self, state):
        """
        Choose action (0: do nothing, 1: jump) based on epsilon-greedy policy
        """
        # Exploration: with epsilon probability, choose a random action
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        
        # Exploitation: choose the best action based on Q-values
        if state not in self.q_table:
            self.q_table[state] = {0: 0, 1: 0}  # Initialize Q-values
            
        q_values = self.q_table[state]
        
        # If both actions have the same value, choose randomly
        if q_values[0] == q_values[1]:
            return random.choice([0, 1])
        
        # Choose the action with highest Q-value
        return max(q_values, key=q_values.get)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value based on reward and next state
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        # Initialize Q-values if not exists
        if state not in self.q_table:
            self.q_table[state] = {0: 0, 1: 0}
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {0: 0, 1: 0}
        
        # Get current Q-value
        q_value = self.q_table[state][action]
        
        # Get max Q-value for next state
        next_max_q = max(self.q_table[next_state].values())
        
        # Update Q-value
        new_q = q_value + self.alpha * (reward + self.gamma * next_max_q - q_value)
        self.q_table[state][action] = new_q
    
    def save_q_table(self, filename="q_table.pkl"):
        """Save Q-table to file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved with {len(self.q_table)} states")
    
    def load_q_table(self, filename="q_table.pkl"):
        """Load Q-table from file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        try:
            with open(filepath, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded with {len(self.q_table)} states")
            return True
        except FileNotFoundError:
            print("No saved Q-table found")
            return False
    
    def report_training_progress(self):
        """Report training progress and estimated time to competency"""
        current_time = pygame.time.get_ticks()
        if current_time - self.last_progress_report < self.report_interval:
            return
            
        # Calculate elapsed time
        elapsed_seconds = (current_time - self.training_start_time) / 1000
        elapsed_minutes = elapsed_seconds / 60
        
        # Calculate states explored per minute
        states_per_minute = len(self.q_table) / max(1, elapsed_minutes)
        
        # Basic estimate: We need roughly 2000-5000 states for decent play
        # More complex estimate would look at state coverage across key areas
        estimated_states_needed = 3000
        remaining_states = max(0, estimated_states_needed - len(self.q_table))
        
        if remaining_states > 0 and states_per_minute > 0:
            estimated_minutes = remaining_states / states_per_minute
            print(f"Training progress: {len(self.q_table)} states explored")
            print(f"Rate: {states_per_minute:.1f} states/minute")
            print(f"Estimated time remaining: {estimated_minutes:.1f} minutes")
        else:
            print(f"Training progress: {len(self.q_table)} states explored")
            if len(self.q_table) > estimated_states_needed:
                print("Sufficient states explored for good gameplay")
            
        self.last_progress_report = current_time


class QBird(Bird):
    def __init__(self, agent):
        """Bird controlled by Q-learning agent"""
        super().__init__()
        self.agent = agent
        self.last_state = None
        self.last_action = None
        self.lifetime = 0
        
    def think(self, pipes):
        """Make a decision based on the environment using Q-learning"""
        # If there are no pipes, don't jump
        if len(pipes) == 0:
            return
        
        # Get the nearest pipe ahead of the bird
        nearest_pipe = None
        nearest_distance = float('inf')
        
        for pipe in pipes:
            # If pipe is ahead of the bird
            if pipe.x + pipe.top_rect.width > self.x:
                distance = pipe.x - self.x
                if distance < nearest_distance:
                    nearest_pipe = pipe
                    nearest_distance = distance
        
        if nearest_pipe is None:
            return
        
        # Get current state
        current_state = self.agent.discretize_state(
            self.y,
            self.velocity,
            nearest_pipe.top_height,
            nearest_distance
        )
        
        # Choose action based on current state
        action = self.agent.get_action(current_state)
        
        # Execute the chosen action
        if action == 1:  # Jump
            self.jump()
        
        # Store current state and action for next update
        self.last_state = current_state
        self.last_action = action


class QGame(Game):
    def __init__(self, training_mode=True, accelerated=False):
        """Game class for Q-learning bird"""
        super().__init__()
        self.agent = QLearningAgent()
        self.agent.load_q_table()  # Try to load existing Q-table
        
        self.bird = QBird(self.agent)
        self.game_started = True
        self.best_score = 0
        self.training_mode = training_mode
        self.current_reward = 0
        self.last_pipe_distance = float('inf')
        self.accelerated = accelerated  # Whether to run in accelerated mode
        
        # Decrease epsilon over time to reduce exploration
        if not training_mode:
            self.agent.epsilon = 0.01  # Less exploration in evaluation mode
        
    def update(self):
        if self.game_over:
            # When game is over, save the Q-table periodically
            if self.agent.training_iterations % 100 == 0:
                self.agent.save_q_table()
                self.agent.report_training_progress()
            self.reset()
            return
            
        # Create new pipes
        time_now = pygame.time.get_ticks()
        if time_now - self.last_pipe > PIPE_FREQUENCY:
            self.pipes.append(Pipe())
            self.last_pipe = time_now
        
        # Get state before bird thinks and moves
        if len(self.pipes) > 0:
            nearest_pipe = None
            nearest_distance = float('inf')
            
            for pipe in self.pipes:
                if pipe.x + pipe.top_rect.width > self.bird.x:
                    distance = pipe.x - self.bird.x
                    if distance < nearest_distance:
                        nearest_pipe = pipe
                        nearest_distance = distance
            
            if nearest_pipe is not None:
                self.last_pipe_distance = nearest_distance
                
                # Calculate reward
                # 1. Small positive reward for staying alive
                reward = 0.1
                
                # 2. Reward for moving toward the center of the pipe gap
                pipe_center_y = nearest_pipe.top_height + PIPE_GAP/2
                bird_distance_to_center = abs(self.bird.y - pipe_center_y)
                normalized_distance = bird_distance_to_center / (SCREEN_HEIGHT/2)
                reward += (1 - normalized_distance) * 0.2  # Max 0.2 for being centered
        
        # Let bird think (select action based on current state)
        self.bird.think(self.pipes)
        self.bird.lifetime += 1
            
        # Update pipes
        for pipe in self.pipes:
            pipe.update()
            
            # Check for collisions
            if pipe.top_rect.colliderect(self.bird.rect) or pipe.bottom_rect.colliderect(self.bird.rect):
                # Large negative reward for collision
                self.current_reward = -10
                self.game_over = True
                
            # Check if pipe has passed the bird
            if pipe.x + PIPE_WIDTH < self.bird.x and not pipe.passed:
                pipe.passed = True
                self.score += 1
                # Large positive reward for passing a pipe
                self.current_reward += 10
                
            # Remove pipes that are off screen
            if pipe.x < -PIPE_WIDTH:
                self.pipes.remove(pipe)
                
        # Update bird
        self.bird.update()
            
        # Ground collision
        if self.bird.y > SCREEN_HEIGHT - GROUND_HEIGHT - GROUND_HEIGHT:
            self.current_reward = -10  # Large negative reward
            self.game_over = True
            
        # Ceiling collision
        if self.bird.y < 0:
            self.current_reward = -10  # Large negative reward
            self.game_over = True
            
        # Q-learning update if we have previous state and action
        if self.training_mode and self.bird.last_state is not None:
            current_state = None
            
            if len(self.pipes) > 0:
                # Find nearest pipe again after updates
                nearest_pipe = None
                nearest_distance = float('inf')
                
                for pipe in self.pipes:
                    if pipe.x + pipe.top_rect.width > self.bird.x:
                        distance = pipe.x - self.bird.x
                        if distance < nearest_distance:
                            nearest_pipe = pipe
                            nearest_distance = distance
                
                if nearest_pipe is not None:
                    current_state = self.agent.discretize_state(
                        self.bird.y,
                        self.bird.velocity,
                        nearest_pipe.top_height,
                        nearest_distance
                    )
                    
                    # Update Q-value
                    self.agent.update_q_value(
                        self.bird.last_state,
                        self.bird.last_action,
                        self.current_reward,
                        current_state
                    )
                    
                    # Reset reward for next step
                    self.current_reward = 0
        
        # Keep track of the best score
        if self.score > self.best_score:
            self.best_score = self.score
            # If we beat our best score, decrease epsilon to exploit more
            if self.agent.epsilon > 0.01:
                self.agent.epsilon *= 0.99
            
        # Update ground scroll
        self.ground_scroll = (self.ground_scroll - PIPE_SPEED) % SCREEN_WIDTH
        
    def reset(self):
        """Reset game for the next attempt"""
        self.agent.training_iterations += 1
        
        # Gradually reduce epsilon (exploration rate) as training progresses
        if self.training_mode and self.agent.epsilon > 0.01:
            self.agent.epsilon *= 0.998  # Slowly decrease exploration
            
        self.bird = QBird(self.agent)
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.last_pipe = pygame.time.get_ticks() - PIPE_FREQUENCY
        self.current_reward = 0
        self.last_pipe_distance = float('inf')
        
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
        screen.blit(self.bird.image, (self.bird.x, self.bird.y))
                
        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        # Draw additional info
        best_score_text = self.font.render(f"Best Score: {self.best_score}", True, WHITE)
        iterations_text = self.font.render(f"Iterations: {self.agent.training_iterations}", True, WHITE)
        states_text = self.font.render(f"Known States: {len(self.agent.q_table)}", True, WHITE)
        epsilon_text = self.font.render(f"Epsilon: {self.agent.epsilon:.3f}", True, WHITE)
        mode_text = self.font.render(f"Mode: {'Training' if self.training_mode else 'Evaluation'}", True, WHITE)
        
        screen.blit(best_score_text, (10, 50))
        screen.blit(iterations_text, (10, 90))
        screen.blit(states_text, (10, 130))
        screen.blit(epsilon_text, (10, 170))
        screen.blit(mode_text, (10, 210))


def main():
    # Import necessary items from main module
    from main import clock
    
    # Initial FPS value
    current_fps = FPS  # Use a local variable to track current FPS
    
    # Create game with Q-learning - default to accelerated training mode
    game = QGame(training_mode=True, accelerated=True)
    running = True
    
    # Display training time estimate
    print("\n=== FLAPPY BIRD Q-LEARNING TRAINING ===")
    print("Estimated training time: 20-40 minutes to reach consistent gameplay")
    print("Tips to speed up training:")
    print("- Press SPACE to increase simulation speed")
    print("- Press A to toggle accelerated mode (less rendering)")
    print("- Press T to toggle between training/evaluation mode")
    print("- Training will automatically save progress every 100 iterations")
    print("=======================================\n")
    
    while running:
        # Skip rendering every other frame in accelerated mode
        if game.accelerated and game.agent.training_iterations % 2 != 0:
            # Process events and update game state without rendering
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game.agent.save_q_table()
                    running = False
            game.update()
            continue
            
        clock.tick(current_fps)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save Q-table before quitting
                game.agent.save_q_table()
                running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    # Save Q-table before quitting
                    game.agent.save_q_table()
                    running = False
                    
                if event.key == pygame.K_SPACE:
                    # Toggle speed
                    if current_fps == 60:
                        current_fps = 120
                    elif current_fps == 120:
                        current_fps = 240
                    else:
                        current_fps = 60

                if event.key == pygame.K_s:
                    # Manual save
                    game.agent.save_q_table("manual_q_table.pkl")
                
                if event.key == pygame.K_t:
                    # Toggle training/evaluation mode
                    game.training_mode = not game.training_mode
                    if game.training_mode:
                        game.agent.epsilon = 0.1  # Higher exploration in training
                    else:
                        game.agent.epsilon = 0.01  # Lower exploration in evaluation
                        
                if event.key == pygame.K_a:
                    # Toggle accelerated mode
                    game.accelerated = not game.accelerated
                    print(f"Accelerated mode: {'ON' if game.accelerated else 'OFF'}")
                    
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
