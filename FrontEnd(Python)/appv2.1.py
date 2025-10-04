import datetime
import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import time
import collections
import heapq
import psutil
import os
import gc

class SlidingPuzzleGame:
    def __init__(self, model_path):
        # Initialize pygame
        pygame.init()
        self.ai_time = 0
        self.search_time = 0
        self.memory_used_mb = 0
        self.search_memory = 0
        self.search_steps = 0
        self.width, self.height = 800, 750  # Increased height for detailed info
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Sliding Puzzle - AI vs Search (Memory Monitor)")
        
        # Colors
        self.BG_COLOR = (240, 240, 240)
        self.TILE_COLOR = (70, 130, 180)
        self.TEXT_COLOR = (255, 255, 255)
        self.EMPTY_COLOR = (200, 200, 200)
        self.BUTTON_COLOR = (100, 150, 200)
        self.PREDICTION_COLOR = (50, 180, 100)
        self.SEARCH_COLOR = (180, 100, 50)
        self.MEMORY_COLOR = (120, 80, 160)
        
        # Game variables
        self.size = 3
        self.tile_size = 100
        self.margin = 5
        self.board_top = 50
        
        # Load AI model
        self.model = load_model(model_path)
        self.model_memory = self.calculate_model_memory()
        self.move_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        
        # Search algorithm tracking
        self.search_method = "BFS"
        self.last_search_path = []
        self.current_search_step = 0
        
        # Memory tracking
        self.process = psutil.Process(os.getpid())
        self.memory_history = []
        self.max_memory_used = 0
        
        # Initialize game
        self.reset_game()

    def calculate_model_memory(self):
        """Calculate the memory usage of the loaded model"""
        try:
            # Get model summary to estimate memory
            model_memory_bytes = 0
            
            # Estimate based on model parameters
            for layer in self.model.layers:
                for weight in layer.get_weights():
                    model_memory_bytes += weight.nbytes
            
            # Add overhead for model structure
            model_memory_bytes += 10 * 1024 * 1024  # 10MB overhead estimate
            
            return model_memory_bytes / (1024 * 1024)  # Convert to MB
        except:
            return 50.0  # Default estimate if calculation fails

    def get_detailed_memory_usage(self):
        """Get detailed memory usage information"""
        try:
            # Process memory
            process_memory = self.process.memory_info()
            rss_mb = process_memory.rss / (1024 * 1024)  # Resident Set Size
            vms_mb = process_memory.vms / (1024 * 1024)  # Virtual Memory Size
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_used_mb = system_memory.used / (1024 * 1024)
            system_available_mb = system_memory.available / (1024 * 1024)
            
            # TensorFlow GPU memory if available
            tf_gpu_memory = 0
            try:
                if tf.config.list_physical_devices('GPU'):
                    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                    tf_gpu_memory = gpu_info['current'] / (1024 * 1024)
            except:
                pass
            
            return {
                'process_rss_mb': rss_mb,
                'process_vms_mb': vms_mb,
                'system_used_mb': system_used_mb,
                'system_available_mb': system_available_mb,
                'tf_gpu_memory_mb': tf_gpu_memory,
                'timestamp': time.time()
            }
        except:
            return {'process_rss_mb': 0, 'process_vms_mb': 0}

    def track_memory_usage(self, operation_name):
        """Track memory usage before and after an operation"""
        # Force garbage collection
        gc.collect()
        
        memory_before = self.get_detailed_memory_usage()
        start_time = time.time()
        
        # This will be replaced with the actual operation
        result = None
        
        # Operation happens here (will be overridden)
        if operation_name == "ai_prediction":
            result = self._ai_prediction_work()
        elif operation_name == "bfs_search":
            result = self._bfs_search_work()
        elif operation_name == "astar_search":
            result = self._astar_search_work()
        
        end_time = time.time()
        
        # Force garbage collection again
        gc.collect()
        
        memory_after = self.get_detailed_memory_usage()
        operation_time = end_time - start_time
        
        # Calculate memory deltas
        memory_delta_rss = memory_after['process_rss_mb'] - memory_before['process_rss_mb']
        memory_delta_vms = memory_after['process_vms_mb'] - memory_before['process_vms_mb']
        
        # Update max memory
        current_rss = memory_after['process_rss_mb']
        if current_rss > self.max_memory_used:
            self.max_memory_used = current_rss
        
        # Store memory history
        self.memory_history.append({
            'operation': operation_name,
            'time': operation_time,
            'memory_delta_rss_mb': memory_delta_rss,
            'memory_delta_vms_mb': memory_delta_vms,
            'memory_before_rss_mb': memory_before['process_rss_mb'],
            'memory_after_rss_mb': memory_after['process_rss_mb'],
            'timestamp': time.time()
        })
        
        return result, {
            'time': operation_time,
            'memory_delta_rss_mb': memory_delta_rss,
            'memory_delta_vms_mb': memory_delta_vms,
            'memory_before_rss_mb': memory_before['process_rss_mb'],
            'memory_after_rss_mb': memory_after['process_rss_mb'],
            'tf_gpu_memory_mb': memory_after['tf_gpu_memory_mb']
        }

    def _ai_prediction_work(self):
        """The actual AI prediction work"""
        flat_board = self.board.flatten()
        one_hot = np.eye(9)[flat_board].flatten().reshape(1, -1)
        pred_probs = self.model.predict(one_hot, verbose=0)
        return pred_probs

    def _bfs_search_work(self):
        """The actual BFS search work"""
        state = self.board.copy()
        start_state = tuple(map(tuple, state))
        goal_state = tuple(map(tuple, self.generate_goal_state()))
        
        if start_state == goal_state:
            return [], 0
        
        queue = collections.deque([(start_state, [])])
        visited = set([start_state])
        states_explored = 0
        
        while queue:
            current_state, path = queue.popleft()
            states_explored += 1
            
            if current_state == goal_state:
                return path, states_explored
            
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
        
        return None, states_explored

    def _astar_search_work(self):
        """The actual A* search work"""
        state = self.board.copy()
        start_state = tuple(map(tuple, state))
        goal_state = tuple(map(tuple, self.generate_goal_state()))
        
        if start_state == goal_state:
            return [], 0
        
        open_set = []
        heapq.heappush(open_set, (0, start_state, [], 0))
        
        g_scores = {start_state: 0}
        f_scores = {start_state: self.manhattan_distance(state)}
        states_explored = 0
        
        while open_set:
            _, current_state, path, g_score = heapq.heappop(open_set)
            states_explored += 1
            
            if current_state == goal_state:
                return path, states_explored
            
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
        
        return None, states_explored

    def get_ai_prediction(self):
        """Get AI prediction with detailed memory tracking"""
        pred_probs, memory_info = self.track_memory_usage("ai_prediction")
        
        self.ai_time = memory_info['time']
        self.memory_used_mb = memory_info['memory_delta_rss_mb']

        predicted_move_index = np.argmax(pred_probs[0])
        predicted_move = self.move_map[predicted_move_index]
        
        self.ai_prediction = {
            'move': predicted_move,
            'probabilities': {
                'up': pred_probs[0][0],
                'down': pred_probs[0][1],
                'left': pred_probs[0][2],
                'right': pred_probs[0][3]
            },
            'memory_info': memory_info
        }
        
        return self.ai_prediction

    def get_search_solution(self):
        """Get solution using search algorithm with memory tracking"""
        if self.search_method == "BFS":
            result, memory_info = self.track_memory_usage("bfs_search")
        else:
            result, memory_info = self.track_memory_usage("astar_search")
        
        path, states_explored = result if result else (None, 0)
        
        self.search_time = memory_info['time']
        self.search_memory = memory_info['memory_delta_rss_mb']
        self.search_steps = states_explored
        
        return path

    # ... (keep the existing methods: reset_game, shuffle_board, make_move, 
    # manhattan_distance, bfs_solve, astar_solve, generate_goal_state, is_solved)
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

    def reset_game(self):
        # Start with goal state
        self.board = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        self.empty_pos = (2, 2)
        self.moves = 0
        self.ai_prediction = None
        self.last_search_path = []
        self.current_search_step = 0
        self.shuffle_board(80)

    def is_solved(self):
        """Check if the puzzle is solved"""
        goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        return np.array_equal(self.board, goal)

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

    def generate_goal_state(self):
        """Generate the goal state"""
        goal = np.arange(1, self.size * self.size + 1).reshape(self.size, self.size)
        goal[-1, -1] = 0
        return goal

    def handle_click(self, pos):
        """Handle mouse clicks"""
        x, y = pos
        
        buttons_top = self.board_top
        button_height = 40
        button_width = 100
        button_spacing = 20
        
        # Row 1 buttons
        if buttons_top <= y <= buttons_top + button_height:
            if 450 <= x <= 550:  # Reset
                self.reset_game()
                return
            elif 570 <= x <= 670:  # AI Move
                prediction = self.get_ai_prediction()
                self.make_move(prediction['move'])
                return
            elif 690 <= x <= 790:  # Shuffle
                self.shuffle_board(10)
                return
        
        # Row 2 buttons
        row2_top = buttons_top + button_height + 10
        if row2_top <= y <= row2_top + button_height:
            if 450 <= x <= 550:  # Search
                path = self.get_search_solution()
                if path:
                    self.last_search_path = path
                    self.current_search_step = 0
                    print(f"Search found solution in {len(path)} moves")
                else:
                    print("No solution found!")
                return
            elif 570 <= x <= 670:  # Toggle search method
                self.search_method = "A*" if self.search_method == "BFS" else "BFS"
                return
            elif 690 <= x <= 790:  # Next search step
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

    def draw_memory_info(self, y_position):
        """Draw detailed memory information"""
        current_memory = self.get_detailed_memory_usage()
        font = pygame.font.Font(None, 20)
        
        # Current memory usage
        memory_y = y_position
        texts = [
            f"Current Memory: {current_memory['process_rss_mb']:.1f} MB (RSS)",
            f"Peak Memory: {self.max_memory_used:.1f} MB",
            f"Model Memory: {self.model_memory:.1f} MB",
            f"System Available: {current_memory['system_available_mb']:.1f} MB"
        ]
        
        for i, text in enumerate(texts):
            memory_text = font.render(text, True, self.MEMORY_COLOR)
            self.screen.blit(memory_text, (20, memory_y + i * 20))
        
        return memory_y + len(texts) * 20 + 10

    def draw(self):
        """Draw the game interface with memory information"""
        self.screen.fill(self.BG_COLOR)
        
        # Draw title
        font = pygame.font.Font(None, 36)
        title = font.render("Sliding Puzzle - Memory Monitor", True, (0, 0, 0))
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 10))
        
        # Draw board
        board_bottom = self.board_top + self.size * (self.tile_size + self.margin)
        for row in range(self.size):
            for col in range(self.size):
                value = self.board[row, col]
                x = col * (self.tile_size + self.margin) + 50
                y = row * (self.tile_size + self.margin) + self.board_top
                
                if value == 0:
                    pygame.draw.rect(self.screen, self.EMPTY_COLOR, (x, y, self.tile_size, self.tile_size))
                else:
                    pygame.draw.rect(self.screen, self.TILE_COLOR, (x, y, self.tile_size, self.tile_size))
                    text = font.render(str(value), True, self.TEXT_COLOR)
                    text_rect = text.get_rect(center=(x + self.tile_size // 2, y + self.tile_size // 2))
                    self.screen.blit(text, text_rect)
        
        # Draw moves counter
        moves_y = board_bottom + 20
        moves_text = font.render(f"Moves: {self.moves}", True, (0, 0, 0))
        self.screen.blit(moves_text, (20, moves_y))
        
        # Draw AI prediction info
        if self.ai_prediction:
            pred_font = pygame.font.Font(None, 20)
            ai_info_y = moves_y + 30
            
            # AI performance
            ai_perf_texts = [
                f"AI Time: {self.ai_time*1000:.2f}ms",
                f"AI Memory: {self.memory_used_mb:.2f} MB",
                f"AI Suggests: {self.ai_prediction['move']}"
            ]
            
            for i, text in enumerate(ai_perf_texts):
                perf_text = pred_font.render(text, True, self.PREDICTION_COLOR)
                self.screen.blit(perf_text, (20, ai_info_y + i * 20))
            
            # AI probabilities
            prob_y = ai_info_y + 70
            for move, prob in self.ai_prediction['probabilities'].items():
                prob_text = pred_font.render(f"{move}: {prob*100:.1f}%", True, (0, 0, 0))
                self.screen.blit(prob_text, (20, prob_y))
                prob_y += 20
        
        # Draw search information
        search_y = board_bottom + 180
        search_font = pygame.font.Font(None, 20)
        search_texts = [
            f"Search Method: {self.search_method}",
            f"Search Time: {self.search_time*1000:.2f}ms" if self.search_time > 0 else "Search Time: --",
            f"Search Memory: {self.search_memory:.2f} MB" if self.search_memory > 0 else "Search Memory: --",
            f"States Explored: {self.search_steps}" if self.search_steps > 0 else "States Explored: --"
        ]
        
        for i, text in enumerate(search_texts):
            search_text = search_font.render(text, True, self.SEARCH_COLOR)
            self.screen.blit(search_text, (20, search_y + i * 20))
        
        # Draw memory information
        memory_start_y = search_y + 100
        final_y = self.draw_memory_info(memory_start_y)
        
        # Draw buttons
        buttons_top = self.board_top + 10
        button_width, button_height = 100, 40
        button_spacing = 20
        
        marg_left = 425

        # Row 1
        reset_rect = pygame.Rect(marg_left, buttons_top, button_width, button_height)
        ai_rect = pygame.Rect(marg_left + button_width + button_spacing, buttons_top, button_width, button_height)
        shuffle_rect = pygame.Rect(marg_left + 2 * (button_width + button_spacing), buttons_top, button_width, button_height)
        
        # Row 2
        search_rect = pygame.Rect(marg_left, buttons_top + button_height + 10, button_width, button_height)
        toggle_rect = pygame.Rect(marg_left + button_width + button_spacing, buttons_top + button_height + 10, button_width, button_height)
        next_rect = pygame.Rect(marg_left + 2 * (button_width + button_spacing), buttons_top + button_height + 10, button_width, button_height)
        
        # Draw buttons
        for rect in [reset_rect, ai_rect, shuffle_rect, search_rect, toggle_rect, next_rect]:
            color = self.SEARCH_COLOR if rect in [search_rect, toggle_rect, next_rect] else self.BUTTON_COLOR
            pygame.draw.rect(self.screen, color, rect)
        
        # Button labels
        button_font = pygame.font.Font(None, 20)
        buttons = [
            (reset_rect, "Reset"),
            (ai_rect, "AI Move"),
            (shuffle_rect, "Shuffle"),
            (search_rect, "Search"),
            (toggle_rect, "Toggle"),
            (next_rect, "Next Step")
        ]
        
        for rect, label in buttons:
            text = button_font.render(label, True, self.TEXT_COLOR)
            self.screen.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
        
        # Draw victory message
        if self.is_solved():
            victory_font = pygame.font.Font(None, 36)
            victory_text = victory_font.render("Puzzle Solved!", True, (0, 150, 0))
            victory_y = buttons_top + 2 * (button_height + 10) + 30
            self.screen.blit(victory_text, (self.width // 2 - victory_text.get_width() // 2, victory_y))

    def handle_clickk(self, pos):
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

if __name__ == "__main__":
    game = SlidingPuzzleGame("puzzle_solver_ann(My).keras")
    game.run()