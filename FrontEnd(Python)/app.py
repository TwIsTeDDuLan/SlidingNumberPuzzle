import datetime
import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import time

class SlidingPuzzleGame:
    def __init__(self, model_path):
        # Initialize pygame
        pygame.init()
        self.ai_time = 0
        self.memory_used_mb = 0
        self.width, self.height = 400, 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Sliding Puzzle")
        
        # Colors
        self.BG_COLOR = (240, 240, 240)
        self.TILE_COLOR = (70, 130, 180)
        self.TEXT_COLOR = (255, 255, 255)
        self.EMPTY_COLOR = (200, 200, 200)
        self.BUTTON_COLOR = (100, 150, 200)
        self.PREDICTION_COLOR = (50, 180, 100)
        
        # Game variables
        self.size = 3
        self.tile_size = 100
        self.margin = 5
        self.board_top = 50
        
        # Load AI model
        self.model = load_model(model_path)
        self.move_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        
        # Initialize game
        self.reset_game()
        
    def reset_game(self):
        # Start with goal state
        self.board = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        self.empty_pos = (2, 2)
        self.moves = 0
        self.ai_prediction = None
        self.shuffle_board(80)  # Shuffle the board
        
    def shuffle_board(self, num_moves):
        """Shuffle the board with random moves"""
        for _ in range(num_moves):
            empty_row, empty_col = self.empty_pos
            possible_moves = []
            
            if empty_row > 0: possible_moves.append('up')
            if empty_row < self.size - 1: possible_moves.append('down')
            if empty_col > 0: possible_moves.append('left')
            if empty_col < self.size - 1: possible_moves.append('right')
            
            if possible_moves:
                nmove = np.random.choice(possible_moves)
                self.make_move(nmove,True)
    
    def make_move(self, move, shuffleMove=False):
        """Make a move on the board"""
        empty_row, empty_col = self.empty_pos
        
        if shuffleMove:
            if move == 'up' and empty_row > 0:
                self.board[empty_row, empty_col], self.board[empty_row - 1, empty_col] = \
                    self.board[empty_row - 1, empty_col], self.board[empty_row, empty_col]
                self.empty_pos = (empty_row - 1, empty_col)
                return True
                
            elif move == 'down' and empty_row < self.size - 1:
                self.board[empty_row, empty_col], self.board[empty_row + 1, empty_col] = \
                    self.board[empty_row + 1, empty_col], self.board[empty_row, empty_col]
                self.empty_pos = (empty_row + 1, empty_col)
                return True
                
            elif move == 'left' and empty_col > 0:
                self.board[empty_row, empty_col], self.board[empty_row, empty_col - 1] = \
                    self.board[empty_row, empty_col - 1], self.board[empty_row, empty_col]
                self.empty_pos = (empty_row, empty_col - 1)
                return True
                
            elif move == 'right' and empty_col < self.size - 1:
                self.board[empty_row, empty_col], self.board[empty_row, empty_col + 1] = \
                    self.board[empty_row, empty_col + 1], self.board[empty_row, empty_col]
                self.empty_pos = (empty_row, empty_col + 1)
                return True
        else:
            if move == 'up' and empty_row > 0:
                self.board[empty_row, empty_col], self.board[empty_row - 1, empty_col] = \
                    self.board[empty_row - 1, empty_col], self.board[empty_row, empty_col]
                self.empty_pos = (empty_row - 1, empty_col)
                self.moves += 1
                return True
                
            elif move == 'down' and empty_row < self.size - 1:
                self.board[empty_row, empty_col], self.board[empty_row + 1, empty_col] = \
                    self.board[empty_row + 1, empty_col], self.board[empty_row, empty_col]
                self.empty_pos = (empty_row + 1, empty_col)
                self.moves += 1
                return True
                
            elif move == 'left' and empty_col > 0:
                self.board[empty_row, empty_col], self.board[empty_row, empty_col - 1] = \
                    self.board[empty_row, empty_col - 1], self.board[empty_row, empty_col]
                self.empty_pos = (empty_row, empty_col - 1)
                self.moves += 1
                return True
                
            elif move == 'right' and empty_col < self.size - 1:
                self.board[empty_row, empty_col], self.board[empty_row, empty_col + 1] = \
                    self.board[empty_row, empty_col + 1], self.board[empty_row, empty_col]
                self.empty_pos = (empty_row, empty_col + 1)
                self.moves += 1
                return True 
        return False
    
    def get_ai_prediction(self):
        """Get AI prediction for the current board state"""

        tf.keras.backend.clear_session()
        initial_allocated = tf.config.experimental.get_memory_info('GPU:0' if tf.test.is_gpu_available() else 'CPU:0')['current']
        initial_allocated_mb = initial_allocated / 1024 / 1024

        # Flatten the board and one-hot encode
        flat_board = self.board.flatten()
        one_hot = np.eye(9)[flat_board].flatten().reshape(1, -1)
        
        # Get prediction
        start_time = time.time()
        pred_probs = self.model.predict(one_hot, verbose=0)
        end_time = time.time()
        self.ai_time = end_time - start_time
        final_allocated = tf.config.experimental.get_memory_info('GPU:0' if tf.test.is_gpu_available() else 'CPU:0')['current']
        final_allocated_mb = final_allocated / 1024 / 1024
        self.memory_used_mb = final_allocated_mb - initial_allocated_mb

        predicted_move_index = np.argmax(pred_probs[0])
        predicted_move = self.move_map[predicted_move_index]
        
        # Store probabilities for display
        self.ai_prediction = {
            'move': predicted_move,
            'probabilities': {
                'up': pred_probs[0][0],
                'down': pred_probs[0][1],
                'left': pred_probs[0][2],
                'right': pred_probs[0][3]
            }
        }
        
        return self.ai_prediction
    
    def is_solved(self):
        """Check if the puzzle is solved"""
        goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        return np.array_equal(self.board, goal)
    
    def draw(self):
        """Draw the game interface"""
        self.screen.fill(self.BG_COLOR)
        
        # Draw title
        font = pygame.font.Font(None, 36)
        title = font.render("Sliding Puzzle", True, (0, 0, 0))
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 10))
        
        # Draw board
        board_bottom = self.board_top + self.size * (self.tile_size + self.margin)
        for row in range(self.size):
            for col in range(self.size):
                value = self.board[row, col]
                x = col * (self.tile_size + self.margin) + 50
                y = row * (self.tile_size + self.margin) + self.board_top
                
                if value == 0:  # Empty tile
                    pygame.draw.rect(self.screen, self.EMPTY_COLOR, 
                                (x, y, self.tile_size, self.tile_size))
                else:
                    pygame.draw.rect(self.screen, self.TILE_COLOR, 
                                (x, y, self.tile_size, self.tile_size))
                    
                    # Draw number
                    font = pygame.font.Font(None, 36)
                    text = font.render(str(value), True, self.TEXT_COLOR)
                    text_rect = text.get_rect(center=(x + self.tile_size // 2, y + self.tile_size // 2))
                    self.screen.blit(text, text_rect)
        
        # Draw moves counter (below the board)
        moves_y = board_bottom + 20
        moves_text = font.render(f"Moves: {self.moves}", True, (0, 0, 0))
        self.screen.blit(moves_text, (20, moves_y))
        
        # Draw AI prediction (to the right of moves counter)
        if self.ai_prediction:
            pred_font = pygame.font.Font(None, 24)
            move_text = pred_font.render(f"AI suggests: {self.ai_prediction['move']}", True, self.PREDICTION_COLOR)
            self.screen.blit(move_text, (200, moves_y))
            move_time = pred_font.render(f"Time: {self.ai_time*1000:.2f}ms", True, self.PREDICTION_COLOR)
            self.screen.blit(move_time,(200, moves_y+20))
            move_memo = pred_font.render(f"Memory: {self.memory_used_mb}MB", True, self.PREDICTION_COLOR)
            self.screen.blit(move_memo,(200, moves_y+40))
            
            # Draw probabilities below the AI suggestion
            prob_y = moves_y + 30
            prob_x = 20  # Left align probabilities
            
            # Draw probabilities in a cleaner format
            prob_title = pred_font.render("Probabilities:", True, (0, 0, 0))
            self.screen.blit(prob_title, (prob_x, prob_y))
            
            prob_y += 25
            for move, prob in self.ai_prediction['probabilities'].items():
                prob_text = pred_font.render(f"  {move}: {prob*100}%", True, (0, 0, 0))
                self.screen.blit(prob_text, (prob_x, prob_y))
                prob_y += 25
        
        # Draw buttons (below everything else)
        buttons_top = board_bottom + 180  # Increased space for probabilities
        button_width, button_height = 100, 40
        button_spacing = 20
        
        # Reset button
        reset_rect = pygame.Rect(50, buttons_top, button_width, button_height)
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, reset_rect)
        
        # AI Move button
        ai_rect = pygame.Rect(50 + button_width + button_spacing, buttons_top, button_width, button_height)
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, ai_rect)
        
        # Shuffle button
        shuffle_rect = pygame.Rect(50 + 2 * (button_width + button_spacing), buttons_top, button_width, button_height)
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, shuffle_rect)
        
        # Button labels
        button_font = pygame.font.Font(None, 24)
        reset_text = button_font.render("Reset", True, self.TEXT_COLOR)
        ai_text = button_font.render("AI Move", True, self.TEXT_COLOR)
        shuffle_text = button_font.render("Shuffle", True, self.TEXT_COLOR)
        
        self.screen.blit(reset_text, (reset_rect.centerx - reset_text.get_width() // 2, 
                                    reset_rect.centery - reset_text.get_height() // 2))
        self.screen.blit(ai_text, (ai_rect.centerx - ai_text.get_width() // 2, 
                                ai_rect.centery - ai_text.get_height() // 2))
        self.screen.blit(shuffle_text, (shuffle_rect.centerx - shuffle_text.get_width() // 2, 
                                    shuffle_rect.centery - shuffle_text.get_height() // 2))
        
        # Draw victory message (centered below buttons)
        if self.is_solved():
            victory_font = pygame.font.Font(None, 48)
            victory_text = victory_font.render("Puzzle Solved!", True, (0, 150, 0))
            victory_y = buttons_top + button_height + 30
            self.screen.blit(victory_text, (self.width // 2 - victory_text.get_width() // 2, victory_y))
    
    def handle_click(self, pos):
        """Handle mouse clicks"""
        x, y = pos
        
        # Button positions (updated to match new layout)
        buttons_top = self.board_top + self.size * (self.tile_size + self.margin) + 180
        button_height = 40
        
        if buttons_top <= y <= buttons_top + button_height:
            button_width = 100
            button_spacing = 20
            
            if 50 <= x <= 150:  # Reset button
                self.reset_game()
                return
            elif 170 <= x <= 270:  # AI Move button (50 + 100 + 20)
                prediction = self.get_ai_prediction()
                self.make_move(prediction['move'])
                return
            elif 290 <= x <= 390:  # Shuffle button (50 + 2*(100 + 20))
                self.shuffle_board(10)
                return
        
        # Check tile clicks for manual moves (unchanged)
        if self.board_top <= y <= self.board_top + self.size * (self.tile_size + self.margin):
            click_row = (y - self.board_top) // (self.tile_size + self.margin)
            click_col = (x - 50) // (self.tile_size + self.margin)
            
            if 0 <= click_row < self.size and 0 <= click_col < self.size:
                empty_row, empty_col = self.empty_pos
                
                # Check if clicked tile is adjacent to empty space
                if (abs(click_row - empty_row) == 1 and click_col == empty_col) or \
                (abs(click_col - empty_col) == 1 and click_row == empty_row):
                    
                    # Determine move direction
                    if click_row == empty_row - 1:
                        self.make_move('up')
                    elif click_row == empty_row + 1:
                        self.make_move('down')
                    elif click_col == empty_col - 1:
                        self.make_move('left')
                    elif click_col == empty_col + 1:
                        self.make_move('right')
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    # Keyboard controls
                    if event.key == pygame.K_UP:
                        self.make_move('up')
                    elif event.key == pygame.K_DOWN:
                        self.make_move('down')
                    elif event.key == pygame.K_LEFT:
                        self.make_move('left')
                    elif event.key == pygame.K_RIGHT:
                        self.make_move('right')
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_a:
                        prediction = self.get_ai_prediction()
                        self.make_move(prediction['move'])
                    elif event.key == pygame.K_s:
                        self.shuffle_board(10)
            
            self.draw()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

# Run the game
if __name__ == "__main__":
    game = SlidingPuzzleGame("puzzle_solver_ann(My).keras")  # Your model file
    game.run()