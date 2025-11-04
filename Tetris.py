import pygame
import random
import torch
from Agent import Agent
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np
import numba



# Epsilon vals
EPSILON_END = 0.001
EPSILON_DECAY = 0.999

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)

# Grid and block sizing
BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
BORDER_WIDTH = 4
SCREEN_WIDTH = BLOCK_SIZE * GRID_WIDTH + BORDER_WIDTH * 2 + 200
SCREEN_HEIGHT = BLOCK_SIZE * GRID_HEIGHT + BORDER_WIDTH

# Tetromino shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I-shape
    [[1, 1], [1, 1]],  # O-shape
    [[1, 1, 1], [0, 1, 0]],  # T-shape
    [[1, 1, 1], [1, 0, 0]],  # L-shape
    [[1, 1, 1], [0, 0, 1]],  # J-shape
    [[1, 1, 0], [0, 1, 1]],  # S-shape
    [[0, 1, 1], [1, 1, 0]]  # Z-shape
]

# Color array
COLORS = [CYAN, YELLOW, MAGENTA, RED, GREEN, BLUE, ORANGE]

# Initialize Pygame
pygame.init()

class TetrisAI:
    """
    Manages the Tetris game logic, state, and rendering for the AI.

    This class handles the game board, piece movement, line clearing,
    and provides methods for the AI agent to get states, simulate moves,
    and play steps.
    """


    def __init__(self):
        """Initializes the Tetris game environment.

        Sets up the pygame display, clock, screen and game grid.
        """
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris")

        self.clock = pygame.time.Clock()

        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        self.next_piece = self.new_piece()

        self.reset()



    def reset(self):
        """Resets the tetris game state."""
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()

        self.game_over = False
        self.score = 0

        self.font = pygame.font.Font(None, 36)

        self.move_delay = 100
        self.last_move_time = {pygame.K_LEFT: 0, pygame.K_RIGHT: 0, pygame.K_DOWN: 0}

        self.frame_iteration = 0

    def new_piece(self):
        """Creates a new Tetromino piece.

        Randomly selects a shape from SHAPES, assigns its corresponding
        color, and sets its initial 'x' and 'y' position at the top
        center of the grid.

        Returns:
            dict: Dictionary representing the new piece, with its shape, color, x, y, and shape_index.
        """
        shape = random.choice(SHAPES)
        color = COLORS[SHAPES.index(shape)]

        shape_index = SHAPES.index(shape)
        color = COLORS[shape_index]

        return {
            'shape': shape,
            'color': color,
            'x': GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0,
            'shape_index': shape_index
        }

    def valid_move(self, piece, x, y):
        """Checks if a pieces move is valid.

        A move is valid if the piece is within grid boundaries and has no collisions

        Args:
            piece (dict): Piece to check.
            x (int): Target x coordinate.
            y (int): Target y coordinate.

        Returns:
            bool: True if move is valid, otherwise false.
        """
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    if (x + j < 0 or x + j >= GRID_WIDTH or
                        y + i >= GRID_HEIGHT or
                        (y + i >= 0 and self.grid[y+i][x+j])):
                        return False
        return True

    def place_piece(self, piece):
        """Confirms placement of a piece in a position

        After placement, sets current piece to next_piece
        and generates a new 'next_piece'.

        Args:
            piece (dict): Piece to place.
        """
        for i, row in enumerate(piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    self.grid[piece['y'] + i][piece['x'] + j] = piece['color']
        # Get next piece
        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()

    def remove_full_rows(self):
        """Checks and removes all completed rows.

        Shifts all rows above the cleared rows down and adds empty rows to the top

        Returns:
            int: Number of rows cleared.
        """
        full_rows = [i for i, row in enumerate(self.grid) if all(row)]
        for row in full_rows:
            del self.grid[row]
            self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
        return len(full_rows)

    def rotate_piece(self, piece):
        """Rotates a piece by 90 degrees.

        Args:
            piece (dict): Piece to rotate.

        Returns:
            dict: A new piece dict with the rotated piece.
        """
        return {
            'shape': list(zip(*reversed(piece['shape']))),
            'color': piece['color'],
            'x': piece['x'],
            'y': piece['y'],
            'shape_index': piece['shape_index']
        }

    def draw_border(self):
        """Draws game border."""
        pygame.draw.rect(self.screen, GRAY, (0, 0, SCREEN_WIDTH - 200, SCREEN_HEIGHT), BORDER_WIDTH)

    def draw(self):
        """Renders the game state to the screen.

        Also renders the 'Game over' message if applicable.
        """
        self.screen.fill(BLACK)

        self.draw_border()
        # Draw existing pieces
        for y, row in enumerate(self.grid):
            for x, color in enumerate(row):
                if color:
                    pygame.draw.rect(self.screen, color,
                                     (x * BLOCK_SIZE + BORDER_WIDTH,
                                      y * BLOCK_SIZE,
                                      BLOCK_SIZE - 1, BLOCK_SIZE - 1))
        # Draw curr piece
        for i, row in enumerate(self.current_piece['shape']):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, self.current_piece['color'],
                                     ((self.current_piece['x'] + j) * BLOCK_SIZE + BORDER_WIDTH,
                                      (self.current_piece['y'] +i) * BLOCK_SIZE,
                                      BLOCK_SIZE - 1, BLOCK_SIZE - 1))


        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (SCREEN_WIDTH - 190, 10))

        if self.game_over:
            game_over_text = self.font.render("Game Over", True, WHITE)
            self.screen.blit(game_over_text, (SCREEN_WIDTH // 2 - 70, SCREEN_HEIGHT // 2))

        pygame.display.flip()

    def _simulate_placement(self, board, piece_to_place):
        """Simulates placement of a piece.

        Helper function to see result of a move without modifying game state.

        Args:
            board (list): Board state.
            piece_to_place (dict): Piece in its final locked position

        Returns:
            tuple: A tuple containing a list with the new board state after placement
            and line clears
        """
        new_board = [row[:] for row in board]

        for i, row in enumerate(piece_to_place['shape']):
            for j,cell in enumerate(row):
                if cell:
                    y_pos, x_pos = piece_to_place['y'] + i, piece_to_place['x'] + j
                    if 0 <= y_pos < GRID_HEIGHT and 0 <= x_pos < GRID_WIDTH:
                        new_board[y_pos][x_pos] = piece_to_place['color']

        lines_cleared = 0

        final_board = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        new_row_index = GRID_HEIGHT - 1

        for y in range(GRID_HEIGHT - 1, -1, -1):
            if not all(new_board[y]):
                final_board[new_row_index] = new_board[y]
                new_row_index -= 1
            else:
                lines_cleared += 1

        return final_board, lines_cleared

    def get_all_possible_next_states(self):
        """Generates all final board states with current piece.

        Iterates through all rotations and horizontal positions for the current piece.
        For each, finds the lowest possible vertical position and sims the placement.

        Returns:
            list: A list of tuples containing the final board states and lines cleared.
        """

        start_piece = self.current_piece
        current_grid = self.grid

        #List to store final moves
        final_moves = []
        final_board_hashes = set()

        # All rotations
        all_rotations = [start_piece['shape']]
        temp_piece = self.rotate_piece(start_piece)
        all_rotations.append(temp_piece['shape'])
        temp_piece = self.rotate_piece(temp_piece)
        all_rotations.append(temp_piece['shape'])
        temp_piece = self.rotate_piece(temp_piece)
        all_rotations.append(temp_piece['shape'])

        # Remove dupes
        unique_shapes = []
        for shape in all_rotations:
            if shape not in unique_shapes:
                unique_shapes.append(shape)

        for shape in unique_shapes:
            for x in range(-2, GRID_WIDTH):
                piece_to_try = {
                    'shape': shape,
                    'color': start_piece['color'],
                    'x': x,
                    'y': 0,
                    'shape_index': start_piece['shape_index']
                }

               # Check if valid for x
                if not self.valid_move(piece_to_try, x, 0):
                    continue

                # Find lowest valid y
                y = 0
                while self.valid_move(piece_to_try, x, y + 1):
                    y += 1

                piece_to_try['y'] = y

                sim_board, lines_cleared = self._simulate_placement(current_grid, piece_to_try)

                board_hash = tuple(tuple(row) for row in sim_board)
                # Add move to final moves if not duplicate
                if board_hash not in final_board_hashes:
                    final_moves.append((sim_board, lines_cleared))
                    final_board_hashes.add(board_hash)

        return final_moves

    @staticmethod
    @numba.njit
    def get_features(board, lines_cleared):
        """Calculates features for a board state.

        Features include aggregate height, number of holes, max height, max well depth and bumpiness.

        Args:
            board (list): Board state.
            lines_cleared (int): Number of lines cleared.

        Returns:
            np.ndarray: A 1D numpy array of the calculated features.
        """
        heights = [0] * GRID_WIDTH
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if board[y][x]:
                    heights[x] = GRID_HEIGHT - y
                    break

        # Calculate agg height
        aggregate_height = sum(heights)
        max_height = max(heights)

        # Calculate holes
        num_holes = 0
        for x in range(GRID_WIDTH):
            col_has_block = False
            for y in range(GRID_HEIGHT):
                if board[y][x]:
                    col_has_block = True
                elif col_has_block and not board[y][x]:
                    num_holes += 1

        #Calculate largest well (lowest column)
        well_depths = [0] * GRID_WIDTH
        for x in range(GRID_WIDTH):
            left_height = heights[x - 1] if x > 0 else max_height
            right_height = heights[x + 1] if x < GRID_WIDTH - 1 else max_height

            cliff_height = min(left_height, right_height)

            if cliff_height > heights[x]:
                well_depths[x] = cliff_height - heights[x]

        # Calculate bumpiness
        bumpiness = 0
        for x in range(GRID_WIDTH - 1):
            bumpiness += abs(heights[x] - heights[x + 1])

        max_well_depth = max(well_depths) if well_depths else 0

        feature_vector = [
            aggregate_height,
            num_holes,
            max_height,
            max_well_depth,
            bumpiness,
            lines_cleared
        ]

        return np.array(feature_vector, dtype=np.float32)

    def calculate_reward(self, board_features, lines_cleared, game_over, holes_before, holes_after):
        """Calculates the rewards for a given move.

        The reward function heavily punishes game over, heavily rewards line clears and applies
        penalty/reward based on the change in number of holes

        Args:
            board_features (torch.Tensor): Features of the resulting board.
            lines_cleared (int): Number of lines cleared.
            game_over (bool): Whether the game is over.
            holes_before (int): Number of holes before the move.
            holes_after (int): Number of holes after the move.

        Returns:
            float: Calculated move reward.
        """
        if game_over:
            return -5000
        # Rewards cooresponding to lines cleared
        # 4 line clear has super high reward to push the agent towards these
        LINE_CLEAR_REWARD = [0, 200, 500, 1000, 250000]

        # Penalties and rewards for holes and placing pieces
        HOLE_PENALTY_WEIGHT = 300
        HOLE_FILL_REWARD_WEIGHT = 100
        STEP_PENALTY = 100

        # Add reward for lines cleared and clearing holes, and remove reward for every piece placed and for increasing holes
        reward = LINE_CLEAR_REWARD[lines_cleared]
        reward -= STEP_PENALTY

        delta_holes = holes_after - holes_before

        if delta_holes > 0:
            reward -= delta_holes * HOLE_PENALTY_WEIGHT
        elif delta_holes < 0:
            reward += abs(delta_holes) * HOLE_FILL_REWARD_WEIGHT

        return reward

    def play_step(self, chosen_move, holes_before):
        """Executes a single game step based on the agent's chosen move

        This function updates the actual game grid with the chosen move, updates the score,
        spawns the next piece, checks for game over and calculates the reward.

        Args:
            chosen_move (tuple): The (sim_board, lines_cleared) tuple chosen by the agent.
            holes_before (int): Number of holes before the move.

        Returns:
            tuple: A tuple containing the reward, whether the game is over and the score.
        """

        self.frame_iteration += 1

        sim_board, lines_cleared = chosen_move
        self.grid = sim_board

        # Add score
        if lines_cleared == 1:
            self.score += 100
        elif lines_cleared == 2:
            self.score += 300
        elif lines_cleared == 3:
            print("Three Lines Cleared!")
            self.score += 500
        elif lines_cleared == 4:
            print("Four Lines Cleared!")
            self.score += 800

        # Get next piece
        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()

        if not self.valid_move(self.current_piece, self.current_piece['x'], self.current_piece['y']):
            self.game_over = True
        else:
            self.game_over = False

        board_features_np = self.get_features(np.array([[1 if cell else 0 for cell in row] for row in self.grid]),
                                              lines_cleared)
        board_features_tensor = torch.from_numpy(board_features_np).unsqueeze(0)
        holes_after = board_features_np[1]
        reward = self.calculate_reward(board_features_tensor, lines_cleared, self.game_over, holes_before, holes_after)
        return reward, self.game_over, self.score


# Training loop
def train():
    """The main training loop for the Tetris AI.

    Handles the game loop, agent-environment interaction, logging to
    TensorBoard, and model saving.
    """
    # Can be toggled with 'H' key
    headless_mode = False

    # Save runs and init training
    writer = SummaryWriter('runs/tetris_run_04')
    total_q_val = 0
    total_steps_for_q = 0
    total_lines = 0
    all_scores = deque(maxlen=100)
    game = TetrisAI()
    agent = Agent()

    global EPSILON_START, EPSILON_END, EPSILON_DECAY
    epsilon = agent.epsilon

    # Update target network every 25 games
    TARGET_UPDATE_FREQUENCY = 25

    running = True

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    headless_mode = not headless_mode
                    print(f"Headless mode: {'ON' if headless_mode else 'OFF'}")
        if not running:
            break

        # Current grid as np array
        current_grid_np = np.array([[1 if cell else 0 for cell in row] for row in game.grid])
        current_features_np = game.get_features(current_grid_np, 0)
        holes_before_move = current_features_np[1]

        # Get piece placements
        placement_moves = game.get_all_possible_next_states()
        if not placement_moves:
            # This should not happen unless spawn is invalid
            game.game_over = True
            print("ERROR: No valid placement moves on spawn.")
            placement_features_list = []
        else:
            features_np_list = [game.get_features(np.array([[1 if cell else 0 for cell in row] for row in b]), l) for (b, l) in placement_moves] # 2. Convert to a list of tensors for the agent
            placement_features_list = [torch.from_numpy(features).unsqueeze(0) for features in features_np_list]



        # Agent chooses action
        if not placement_features_list:
            # Game over
            game.game_over = True
            reward = -5000
            done = True
            score = game.score

            agent.remember(torch.zeros(1, 4), reward, [], done)
            state_to_remember = torch.zeros(1, 4)  # Need this for the 'remember' call later

        else:
            # Ask the agent for its decision
            action_type, chosen_idx, chosen_features, q_pred = agent.get_action(
                placement_features_list,
                epsilon
            )

            # Accumulate Q Value
            q_value_this_step = q_pred.item()
            total_q_val += q_value_this_step
            total_steps_for_q += 1


            # Chosen move
            chosen_move = placement_moves[chosen_idx]

            # Lines cleared now and total
            lines_cleared_this_step = chosen_move[1]
            total_lines += lines_cleared_this_step

            # Input chosen move to play_step function
            reward, done, score = game.play_step(chosen_move, holes_before_move)
            state_to_remember = chosen_features


            # Get next state
            next_placement_features = []
            if not done:
                # Get all possible moves for the new piece
                next_moves = game.get_all_possible_next_states()
                if next_moves:
                    next_features_np = [game.get_features(np.array([[1 if cell else 0 for cell in row] for row in b]), l) for (b, l) in next_moves]
                    next_placement_features = [torch.from_numpy(features).unsqueeze(0) for features in next_features_np]
                else:
                    # Game is over next step
                    done = True


            # We store the features of the move we took, the reward we got, the list of all features for the next state, and 'done'.
        agent.remember(state_to_remember, reward, next_placement_features, done)
        loss = agent.train_long_memory()

        # Render if not headless
        if not headless_mode:
            game.draw()
            # Adjust to make game render quicker/slower
            pygame.time.wait(2)

        # Handle game over, saving and storing data to tensorboard
        if done:
            current_game_steps = game.frame_iteration
            all_scores.append(score)
            avg_score = sum(all_scores) / len(all_scores)
            if agent.n_games % 50 == 0:
                checkpoint_name = f'tetris_game_{agent.n_games}.pth'
                agent.save_model(checkpoint_name)
                agent.save_state()
            game.reset()
            agent.n_games += 1

            # Update the target
            if agent.n_games % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()
                print(f"Game {agent.n_games}: Target network updated.")

            # Decay epsilon
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            agent.epsilon = epsilon
            agent.save_state()

            # Print game, score, avg score and epsilon
            print(f"Game: {agent.n_games}, Score: {score}, Avg Score: {avg_score:.2f}, Epsilon: {epsilon:.4f}")

            # Log per game stats
            writer.add_scalar('Game/Score', score, agent.n_games)
            writer.add_scalar('Game/Lines_Cleared', total_lines, agent.n_games)
            writer.add_scalar('Game/Steps_Survived', current_game_steps, agent.n_games)
            writer.add_scalar('Game/Avg_Score_100', avg_score, agent.n_games)
            writer.add_scalar('Training/Epsilon', epsilon, agent.n_games)

            # Log avg Q-Value
            if total_steps_for_q > 0:
                avg_q = total_q_val / total_steps_for_q
                writer.add_scalar('Network/Avg_Q_Per_Game', avg_q, agent.n_games)

            # Reset per game tracker
            total_q_val = 0
            total_steps_for_q = 0
            total_lines = 0


    # End of while
    writer.close()
    pygame.quit()
    quit()


if __name__ == "__main__":
    train()
