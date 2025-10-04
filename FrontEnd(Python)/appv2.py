import datetime
import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import time
import collections
import heapq

class SlidingPuzzleGame:
    def __init__(self, model_path):
        # Initialize pygame
        pygame.init()
        self.ai_time = 0
        self.search_time = 0
        self.memory_used_mb = 0
        self.search_memory = 0
        self.search_steps = 0
        self.width, self.height = 400, 750  # Increased height for search info
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Sliding Puzzle - AI vs Search")
        
        # Colors
        self.BG_COLOR = (240, 240, 240)
        self.TILE_COLOR = (70, 130, 180)
        self.TEXT_COLOR = (255, 255, 255)
        self.EMPTY_COLOR = (200, 200, 200)
        self.BUTTON_COLOR = (100, 150, 200)
        self.PREDICTION_COLOR = (50, 180, 100)
        self.SEARCH_COLOR = (180, 100, 50)
        
        # Game variables
        self.size = 3
        self.tile_size = 100
        self.margin = 5
        self.board_top = 50
        
        # Load AI model
        self.model = load_model(model_path)
        self.move_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        
        # Search algorithm tracking
        self.search_method = "BFS"  # Default search method
        self.last_search_path = []
        self.current_search_step = 0
        
        # Initialize game
        self.reset_game()
        
    def reset_game(self):
        # Start with goal state
        self.board = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        self.empty_pos = (2, 2)
        self.moves = 0
        self.ai_prediction = None
        self.last_search_path = []
        self.current_search_step = 0
        self.shuffle_board(80)
        
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
                self.make_move(nmove, True)
    
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

    def manhattan_distance(self, state):
        """Calculate Manhattan distance heuristic for A* search"""
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                val = state[i][j]
                if val != 0:
                    goal_row, goal_col = (val - 1) // self.size, (val - 1) % self.size
                    distance += abs(i - goal_row) + abs(j - goal_col)
        return distance

    def bfs_solve(self, state):
        """BFS search to find optimal solution"""
        start_time = time.time()
        start_state = tuple(map(tuple, state))
        goal_state = tuple(map(tuple, self.generate_goal_state()))
        
        if start_state == goal_state:
            return [], 0, 0
        
        queue = collections.deque([(start_state, [])])
        visited = set([start_state])
        states_explored = 0
        
        while queue:
            current_state, path = queue.popleft()
            states_explored += 1
            
            if current_state == goal_state:
                search_time = time.time() - start_time
                return path, states_explored, search_time
            
            # Convert back to numpy for move generation
            state_array = np.array(current_state)
            empty_pos = np.where(state_array == 0)
            empty_row, empty_col = empty_pos[0][0], empty_pos[1][0]
            
            for move, (dr, dc) in [('up', (-1, 0)), ('down', (1, 0)), 
                                  ('left', (0, -1)), ('right', (0, 1))]:
                new_row, new_col = empty_row + dr, empty_col + dc
                
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    new_state = state_array.copy()
                    new_state[empty_row, empty_col], new_state[new_row, new_col] = \
                        new_state[new_row, new_col], new_state[empty_row, empty_col]
                    
                    new_state_tuple = tuple(map(tuple, new_state))
                    
                    if new_state_tuple not in visited:
                        visited.add(new_state_tuple)
                        queue.append((new_state_tuple, path + [move]))
        
        return None, states_explored, time.time() - start_time

    def astar_solve(self, state):
        """A* search with Manhattan distance heuristic"""
        start_time = time.time()
        start_state = tuple(map(tuple, state))
        goal_state = tuple(map(tuple, self.generate_goal_state()))
        
        if start_state == goal_state:
            return [], 0, 0
        
        # Priority queue: (f_score, state, path, g_score)
        open_set = []
        heapq.heappush(open_set, (0, start_state, [], 0))
        
        g_scores = {start_state: 0}
        f_scores = {start_state: self.manhattan_distance(state)}
        states_explored = 0
        
        while open_set:
            _, current_state, path, g_score = heapq.heappop(open_set)
            states_explored += 1
            
            if current_state == goal_state:
                search_time = time.time() - start_time
                return path, states_explored, search_time
            
            state_array = np.array(current_state)
            empty_pos = np.where(state_array == 0)
            empty_row, empty_col = empty_pos[0][0], empty_pos[1][0]
            
            for move, (dr, dc) in [('up', (-1, 0)), ('down', (1, 0)), 
                                  ('left', (0, -1)), ('right', (0, 1))]:
                new_row, new_col = empty_row + dr, empty_col + dc
                
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    new_state = state_array.copy()
                    new_state[empty_row, empty_col], new_state[new_row, new_col] = \
                        new_state[new_row, new_col], new_state[empty_row, empty_col]
                    
                    new_state_tuple = tuple(map(tuple, new_state))
                    tentative_g_score = g_score + 1
                    
                    if new_state_tuple not in g_scores or tentative_g_score < g_scores[new_state_tuple]:
                        g_scores[new_state_tuple] = tentative_g_score
                        f_score = tentative_g_score + self.manhattan_distance(new_state)
                        f_scores[new_state_tuple] = f_score
                        heapq.heappush(open_set, (f_score, new_state_tuple, path + [move], tentative_g_score))
        
        return None, states_explored, time.time() - start_time

    def get_search_solution(self):
        """Get solution using search algorithm"""
        start_memory = self.get_current_memory()
        start_time = time.time()
        
        if self.search_method == "BFS":
            path, states_explored, search_time = self.bfs_solve(self.board.copy())
        else:  # A*
            path, states_explored, search_time = self.astar_solve(self.board.copy())
        
        end_memory = self.get_current_memory()
        self.search_memory = end_memory - start_memory
        self.search_time = search_time
        self.search_steps = states_explored
        
        return path

    def get_current_memory(self):
        """Get current memory usage"""
        tf.keras.backend.clear_session()
        memory_info = tf.config.experimental.get_memory_info('GPU:0' if tf.test.is_gpu_available() else 'CPU:0')
        return memory_info['current'] / 1024 / 1024

    def generate_goal_state(self):
        """Generate the goal state"""
        goal = np.arange(1, self.size * self.size + 1).reshape(self.size, self.size)
        goal[-1, -1] = 0
        return goal

    def get_ai_prediction(self):
        """Get AI prediction for the current board state"""
        tf.keras.backend.clear_session()
        initial_allocated = self.get_current_memory()

        # Flatten the board and one-hot encode
        flat_board = self.board.flatten()
        one_hot = np.eye(9)[flat_board].flatten().reshape(1, -1)
        
        # Get prediction
        start_time = time.time()
        pred_probs = self.model.predict(one_hot, verbose=0)
        end_time = time.time()
        self.ai_time = end_time - start_time
        
        final_allocated = self.get_current_memory()
        self.memory_used_mb = final_allocated - initial_allocated

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
        title = font.render("Sliding Puzzle - AI vs Search", True, (0, 0, 0))
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
        
        # Draw moves counter
        moves_y = board_bottom + 20
        moves_text = font.render(f"Moves: {self.moves}", True, (0, 0, 0))
        self.screen.blit(moves_text, (20, moves_y))
        
        # Draw AI prediction
        if self.ai_prediction:
            pred_font = pygame.font.Font(None, 24)
            move_text = pred_font.render(f"AI suggests: {self.ai_prediction['move']}", True, self.PREDICTION_COLOR)
            self.screen.blit(move_text, (200, moves_y))
            move_time = pred_font.render(f"Time: {self.ai_time*1000:.2f}ms", True, self.PREDICTION_COLOR)
            self.screen.blit(move_time, (200, moves_y+20))
            move_memo = pred_font.render(f"Memory: {self.memory_used_mb:.2f}MB", True, self.PREDICTION_COLOR)
            self.screen.blit(move_memo, (200, moves_y+40))
            
            # Draw probabilities
            prob_y = moves_y + 70
            prob_x = 20
            prob_title = pred_font.render("AI Probabilities:", True, (0, 0, 0))
            self.screen.blit(prob_title, (prob_x, prob_y))
            
            prob_y += 25
            for move, prob in self.ai_prediction['probabilities'].items():
                prob_text = pred_font.render(f"  {move}: {prob*100:.1f}%", True, (0, 0, 0))
                self.screen.blit(prob_text, (prob_x, prob_y))
                prob_y += 25
        
        # Draw search information
        search_y = board_bottom + 180
        search_font = pygame.font.Font(None, 24)
        search_text = search_font.render(f"Search Method: {self.search_method}", True, self.SEARCH_COLOR)
        self.screen.blit(search_text, (20, search_y))
        
        if self.search_time > 0:
            search_time_text = search_font.render(f"Search Time: {self.search_time*1000:.2f}ms", True, self.SEARCH_COLOR)
            self.screen.blit(search_time_text, (20, search_y + 25))
            search_mem_text = search_font.render(f"Search Memory: {self.search_memory:.2f}MB", True, self.SEARCH_COLOR)
            self.screen.blit(search_mem_text, (20, search_y + 50))
            search_steps_text = search_font.render(f"States Explored: {self.search_steps}", True, self.SEARCH_COLOR)
            self.screen.blit(search_steps_text, (20, search_y + 75))
        
        # Draw buttons
        buttons_top = board_bottom + 270
        button_width, button_height = 100, 40
        button_spacing = 20
        
        # Row 1: Game control buttons
        reset_rect = pygame.Rect(50, buttons_top, button_width, button_height)
        ai_rect = pygame.Rect(50 + button_width + button_spacing, buttons_top, button_width, button_height)
        shuffle_rect = pygame.Rect(50 + 2 * (button_width + button_spacing), buttons_top, button_width, button_height)
        
        # Row 2: Search control buttons
        search_rect = pygame.Rect(50, buttons_top + button_height + 10, button_width, button_height)
        toggle_rect = pygame.Rect(50 + button_width + button_spacing, buttons_top + button_height + 10, button_width, button_height)
        next_rect = pygame.Rect(50 + 2 * (button_width + button_spacing), buttons_top + button_height + 10, button_width, button_height)
        
        # Draw buttons
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, reset_rect)
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, ai_rect)
        pygame.draw.rect(self.screen, self.BUTTON_COLOR, shuffle_rect)
        pygame.draw.rect(self.screen, self.SEARCH_COLOR, search_rect)
        pygame.draw.rect(self.screen, self.SEARCH_COLOR, toggle_rect)
        pygame.draw.rect(self.screen, self.SEARCH_COLOR, next_rect)
        
        # Button labels
        button_font = pygame.font.Font(None, 24)
        self.screen.blit(button_font.render("Reset", True, self.TEXT_COLOR), 
                        (reset_rect.centerx - 25, reset_rect.centery - 10))
        self.screen.blit(button_font.render("AI Move", True, self.TEXT_COLOR), 
                        (ai_rect.centerx - 30, ai_rect.centery - 10))
        self.screen.blit(button_font.render("Shuffle", True, self.TEXT_COLOR), 
                        (shuffle_rect.centerx - 30, shuffle_rect.centery - 10))
        self.screen.blit(button_font.render("Search", True, self.TEXT_COLOR), 
                        (search_rect.centerx - 25, search_rect.centery - 10))
        self.screen.blit(button_font.render(f"Toggle", True, self.TEXT_COLOR), 
                        (toggle_rect.centerx - 25, toggle_rect.centery - 10))
        self.screen.blit(button_font.render("Next Step", True, self.TEXT_COLOR), 
                        (next_rect.centerx - 35, next_rect.centery - 10))
        
        # Draw victory message
        if self.is_solved():
            victory_font = pygame.font.Font(None, 48)
            victory_text = victory_font.render("Puzzle Solved!", True, (0, 150, 0))
            victory_y = buttons_top + 2 * (button_height + 10) + 30
            self.screen.blit(victory_text, (self.width // 2 - victory_text.get_width() // 2, victory_y))
    
    def handle_click(self, pos):
        """Handle mouse clicks"""
        x, y = pos
        
        buttons_top = self.board_top + self.size * (self.tile_size + self.margin) + 270
        button_height = 40
        button_width = 100
        button_spacing = 20
        
        # Row 1 buttons
        if buttons_top <= y <= buttons_top + button_height:
            if 50 <= x <= 150:  # Reset
                self.reset_game()
                return
            elif 170 <= x <= 270:  # AI Move
                prediction = self.get_ai_prediction()
                self.make_move(prediction['move'])
                return
            elif 290 <= x <= 390:  # Shuffle
                self.shuffle_board(10)
                return
        
        # Row 2 buttons
        row2_top = buttons_top + button_height + 10
        if row2_top <= y <= row2_top + button_height:
            if 50 <= x <= 150:  # Search
                path = self.get_search_solution()
                if path:
                    self.last_search_path = path
                    self.current_search_step = 0
                    print(f"Search found solution in {len(path)} moves")
                else:
                    print("No solution found!")
                return
            elif 170 <= x <= 270:  # Toggle search method
                self.search_method = "A*" if self.search_method == "BFS" else "BFS"
                return
            elif 290 <= x <= 390:  # Next search step
                if self.last_search_path and self.current_search_step < len(self.last_search_path):
                    move = self.last_search_path[self.current_search_step]
                    self.make_move(move)
                    self.current_search_step += 1
                return
        
        # Tile clicks for manual moves
        if self.board_top <= y <= self.board_top + self.size * (self.tile_size + self.margin):
            click_row = (y - self.board_top) // (self.tile_size + self.margin)
            click_col = (x - 50) // (self.tile_size + self.margin)
            
            if 0 <= click_row < self.size and 0 <= click_col < self.size:
                empty_row, empty_col = self.empty_pos
                
                if (abs(click_row - empty_row) == 1 and click_col == empty_col) or \
                   (abs(click_col - empty_col) == 1 and click_row == empty_row):
                    
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
                    elif event.key == pygame.K_d:  # Search solution
                        path = self.get_search_solution()
                        if path:
                            self.last_search_path = path
                            self.current_search_step = 0
                    elif event.key == pygame.K_t:  # Toggle search method
                        self.search_method = "A*" if self.search_method == "BFS" else "BFS"
                    elif event.key == pygame.K_n:  # Next search step
                        if self.last_search_path and self.current_search_step < len(self.last_search_path):
                            move = self.last_search_path[self.current_search_step]
                            self.make_move(move)
                            self.current_search_step += 1
            
            self.draw()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

# Run the game
if __name__ == "__main__":
    game = SlidingPuzzleGame("puzzle_solver_ann(My).keras")
    game.run()