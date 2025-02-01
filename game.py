import pygame
import sys
import random
import numpy as np
from typing import List


# 1) Environment / State Logic

class MultiRobotSolarFarmEnv:
    """
    Indie-style environment with:
      - Multiple robots
      - Resources, obstacles, panels
      - Panel synergy: each panel produces +1 bonus if adjacent to at least one other panel
      - Score increments each step from all panels
    """

    def __init__(self, grid_size=8, max_steps=300):
        self.grid_size = grid_size
        self.max_steps = max_steps

        # We'll store multiple robots as a list of (x, y)
        self.robot_positions: List[List[int]] = []

        # Maps
        self.obstacle_map = np.zeros((grid_size, grid_size), dtype=int)
        self.resource_map = np.zeros((grid_size, grid_size), dtype=int)
        self.panel_map = np.zeros((grid_size, grid_size), dtype=int)

        self.score = 0.0
        self.step_count = 0

        self.neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # For adjacency synergy

    def reset(self):
        """Random reset for demonstration; you could do something more sophisticated."""
        self.step_count = 0
        self.score = 0

        # Clear maps
        self.obstacle_map[:] = 0
        self.resource_map[:] = 0
        self.panel_map[:] = 0
        self.robot_positions = []

        # Place random robots
        for _ in range(2):  # start with 2 robots
            rx = random.randint(0, self.grid_size - 1)
            ry = random.randint(0, self.grid_size - 1)
            self.robot_positions.append([rx, ry])

        # Place random obstacles
        num_obstacles = random.randint(self.grid_size // 3, self.grid_size // 2)
        for _ in range(num_obstacles):
            ox = random.randint(0, self.grid_size - 1)
            oy = random.randint(0, self.grid_size - 1)
            self.obstacle_map[ox, oy] = 1

        # Place random resources
        num_resources = random.randint(5, 8)
        for _ in range(num_resources):
            rx = random.randint(0, self.grid_size - 1)
            ry = random.randint(0, self.grid_size - 1)
            self.resource_map[rx, ry] = 1

        return self._get_observation()

    def step(self, actions):
        """
        Step the environment by applying a list of actions (one for each robot):
          Actions: 0=Up,1=Right,2=Down,3=Left,4=BuildPanel
          (We do not enforce resource cost in this example, but you could.)
        """
        self.step_count += 1

        # Move each robot according to its action
        for i, action in enumerate(actions):
            if i >= len(self.robot_positions):
                break
            old_x, old_y = self.robot_positions[i]
            new_x, new_y = old_x, old_y

            if action == 0:  # up
                new_x = max(0, old_x - 1)
            elif action == 1:  # right
                new_y = min(self.grid_size - 1, old_y + 1)
            elif action == 2:  # down
                new_x = min(self.grid_size - 1, old_x + 1)
            elif action == 3:  # left
                new_y = max(0, old_y - 1)
            elif action == 4:  # build panel
                # Build a panel if there's no obstacle/panel
                if self.obstacle_map[old_x, old_y] == 0:
                    if self.panel_map[old_x, old_y] == 0:
                        self.panel_map[old_x, old_y] = 1
                # No movement if building
            # else: invalid => ignore

            # If not building, attempt to move
            if action in [0, 1, 2, 3]:
                # If obstacle => revert
                if self.obstacle_map[new_x, new_y] == 1:
                    new_x, new_y = old_x, old_y
                self.robot_positions[i] = [new_x, new_y]

            # If there's a resource, pick it up => add small score for demonstration
            if self.resource_map[new_x, new_y] == 1:
                self.resource_map[new_x, new_y] = 0
                self.score += 1

        # Panel synergy: each panel yields 1 + bonus if at least 1 neighbor is also a panel
        electricity_this_step = 0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.panel_map[x, y] == 1:
                    base = 1
                    bonus = 0
                    for dx, dy in self.neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if self.panel_map[nx, ny] == 1:
                                bonus = 1
                                break
                    electricity_this_step += (base + bonus)

        self.score += electricity_this_step * 0.1

        done = (self.step_count >= self.max_steps)
        obs = self._get_observation()
        return obs, (electricity_this_step * 0.1), done

    def _get_observation(self):
        """Simple flattened observation for demonstration."""
        # e.g., flatten all maps + robot positions + step_count, etc.
        # (We won't use it seriously for the mock agent.)
        flat_obs = []

        # Robot positions
        for rx, ry in self.robot_positions:
            flat_obs.extend([rx, ry])
        # Pad if fewer than e.g. 4 robots
        while len(flat_obs) < 8:
            flat_obs.append(0)

        # Flatten maps
        flat_obs.extend(self.obstacle_map.flatten().tolist())
        flat_obs.extend(self.resource_map.flatten().tolist())
        flat_obs.extend(self.panel_map.flatten().tolist())

        # add step count, score
        flat_obs.append(self.step_count)
        flat_obs.append(self.score)
        return np.array(flat_obs, dtype=float)

    def in_bounds(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def add_robot(self, x, y):
        """Add a new robot at (x, y)."""
        self.robot_positions.append([x, y])



# 2) A Simple (Placeholder) Multi-Robot Agent
#    (Replace with your trained Stable-Baselines3 agent if you want)

class MultiRobotRandomAgent:
    def __init__(self, n_robots=2):
        self.n_robots = n_robots

    def predict(self, obs):
        # For demonstration, pick random action [0..4] for each robot
        actions = [random.randint(0, 4) for _ in range(self.n_robots)]
        return actions, None



# 3) Pygame Interactive "Indie" Game

def run_interactive_game():
    pygame.init()
    cell_size = 64
    grid_size = 8

    screen_width = grid_size * cell_size
    screen_height = grid_size * cell_size + 80  # extra space for text
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Interactive Solar Farm (Indie)")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Load or fallback images
    try:
        img_robot = pygame.image.load("robot.png")
        img_robot = pygame.transform.scale(img_robot, (cell_size, cell_size))
        img_obstacle = pygame.image.load("obstacle.png")
        img_obstacle = pygame.transform.scale(img_obstacle, (cell_size, cell_size))
        img_resource = pygame.image.load("resource.png")
        img_resource = pygame.transform.scale(img_resource, (cell_size, cell_size))
        img_panel = pygame.image.load("panel.png")
        img_panel = pygame.transform.scale(img_panel, (cell_size, cell_size))
        img_sand = pygame.image.load("sand.png")
        img_sand = pygame.transform.scale(img_sand, (cell_size, cell_size))
    except:
        print("Warning: Some sprite images not found. Using colored rectangles.")
        img_robot = None
        img_obstacle = None
        img_resource = None
        img_panel = None
        img_sand = None

    # Create environment and agent
    env = MultiRobotSolarFarmEnv(grid_size=grid_size, max_steps=300)
    obs = env.reset()

    # Suppose we have 2 robots initially
    agent = MultiRobotRandomAgent(n_robots=2)

    mode = "play"  # either "play" or "edit"

    # Colors
    COLOR_BACKGROUND = (30, 30, 30)
    COLOR_GRID = (200, 200, 200)
    COLOR_ROBOT = (255, 100, 100)
    COLOR_OBSTACLE = (70, 70, 70)
    COLOR_RESOURCE = (0, 255, 0)
    COLOR_PANEL = (255, 255, 100)
    COLOR_SAND = (180, 170, 120)
    COLOR_TEXT = (255, 255, 255)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    mode = "edit" if mode == "play" else "play"

            elif event.type == pygame.MOUSEBUTTONDOWN and mode == "edit":
                mx, my = pygame.mouse.get_pos()
                gx = my // cell_size  # row
                gy = mx // cell_size  # col
                if env.in_bounds(gx, gy):
                    # Let's define:
                    #  - Left-click => place Robot
                    #  - Right-click => toggle obstacle
                    #  - Middle-click => place Resource
                    #  - Ctrl+Left => place Panel
                    #  (Feel free to change the logic to your liking!)
                    left_click = (event.button == 1 and not pygame.key.get_mods() & pygame.KMOD_CTRL)
                    ctrl_left_click = (event.button == 1 and (pygame.key.get_mods() & pygame.KMOD_CTRL))
                    right_click = (event.button == 3)
                    middle_click = (event.button == 2) or (
                                pygame.key.get_mods() & pygame.KMOD_SHIFT and event.button == 1)

                    if left_click:
                        # Place robot
                        env.add_robot(gx, gy)
                    elif right_click:
                        # Toggle obstacle
                        if env.obstacle_map[gx, gy] == 1:
                            env.obstacle_map[gx, gy] = 0
                        else:
                            env.obstacle_map[gx, gy] = 1
                        # Clear panel/resource if any
                        env.panel_map[gx, gy] = 0
                        env.resource_map[gx, gy] = 0
                    elif middle_click:
                        # Place resource
                        env.resource_map[gx, gy] = 1
                        env.obstacle_map[gx, gy] = 0
                        env.panel_map[gx, gy] = 0
                    elif ctrl_left_click:
                        # Place panel
                        env.panel_map[gx, gy] = 1
                        env.obstacle_map[gx, gy] = 0
                        env.resource_map[gx, gy] = 0

        if mode == "play":
            # Let the agent act
            actions, _ = agent.predict(obs)
            obs, reward, done = env.step(actions)
            if done:
                obs = env.reset()

        # ----- Render -----
        screen.fill(COLOR_BACKGROUND)

        # Draw grid
        for x in range(grid_size):
            for y in range(grid_size):
                rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)

                # base terrain
                if img_sand is not None:
                    screen.blit(img_sand, rect)
                else:
                    pygame.draw.rect(screen, COLOR_SAND, rect)

                # obstacle
                if env.obstacle_map[x, y] == 1:
                    if img_obstacle:
                        screen.blit(img_obstacle, rect)
                    else:
                        pygame.draw.rect(screen, COLOR_OBSTACLE, rect)

                # resource
                elif env.resource_map[x, y] == 1:
                    if img_resource:
                        screen.blit(img_resource, rect)
                    else:
                        pygame.draw.rect(screen, COLOR_RESOURCE, rect)

                # panel
                if env.panel_map[x, y] == 1:
                    if img_panel:
                        screen.blit(img_panel, rect)
                    else:
                        pygame.draw.rect(screen, COLOR_PANEL, rect)

        # Draw robots
        for (rx, ry) in env.robot_positions:
            rrect = pygame.Rect(ry * cell_size, rx * cell_size, cell_size, cell_size)
            if img_robot:
                screen.blit(img_robot, rrect)
            else:
                pygame.draw.rect(screen, COLOR_ROBOT, rrect)

        # Grid lines
        for x in range(grid_size):
            for y in range(grid_size):
                rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, COLOR_GRID, rect, 1)

        # Info text
        mode_text = f"MODE: {mode.upper()} (Space = toggle)"
        step_text = f"STEP: {env.step_count}/{env.max_steps}"
        score_text = f"SCORE: {env.score:.1f}"
        text_surface = font.render(f"{mode_text} | {step_text} | {score_text} | [ESC] quit", True, COLOR_TEXT)
        screen.blit(text_surface, (10, grid_size * cell_size + 10))

        pygame.display.flip()
        clock.tick(8)  # ~8 FPS for visibility

    pygame.quit()


# 4) Run the Game
if __name__ == "__main__":
    run_interactive_game()
