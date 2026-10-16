"""SOLAR GROVE: pixel-art expedition viewer for PPO and baseline controllers."""
import argparse
import math
import os
from pathlib import Path

import pygame

from agents import GreedyAgent, RandomAgent
from solar_env import ACTION_NAMES, MultiRobotSolarFarmEnv

INK = (24, 37, 43)
PAPER = (240, 230, 202)
MUTED = (158, 174, 160)
GOLD = (244, 189, 85)
GREEN = (114, 185, 142)
BLUE = (118, 187, 205)
W, H = 512, 352
CELL, BX, BY = 28, 16, 64


def rect(surface, color, box, border=0):
    pygame.draw.rect(surface, color, box, border)


def text(surface, value, x, y, color=PAPER, size=12):
    font = pygame.font.Font(pygame.font.match_font('menlo,monospace'), size)
    surface.blit(font.render(str(value), False, color), (x, y))


def panel(surface, box):
    x, y, w, h = box
    rect(surface, (12, 24, 31), (x + 3, y + 3, w, h))
    rect(surface, (35, 53, 57), box)
    rect(surface, (74, 94, 86), box, 1)


def draw_robot(surface, x, y, index, tick):
    accent = BLUE if index == 0 else (232, 145, 111)
    bob = int(math.sin(tick * 0.12 + index) > 0)
    rect(surface, (105, 95, 66), (x + 5, y + 22, 19, 3))
    for dx in (7, 18):
        rect(surface, INK, (x + dx, y + 20, 4, 5))
    rect(surface, INK, (x + 5, y + 6 - bob, 20, 16))
    rect(surface, accent, (x + 6, y + 5 - bob, 18, 14))
    rect(surface, PAPER, (x + 8, y + 6 - bob, 14, 2))
    rect(surface, INK, (x + 9, y + 10 - bob, 12, 5))
    for dx in (11, 18):
        rect(surface, (190, 240, 201), (x + dx, y + 11 - bob, 2, 2))
    rect(surface, INK, (x + 14, y + 1 - bob, 2, 4))
    rect(surface, GOLD, (x + 13, y - bob, 4, 3))


def draw_world(surface, env, tick):
    panel(surface, (12, 60, 232, 232))
    for r in range(8):
        for c in range(8):
            x, y = BX + c * CELL, BY + r * CELL
            base = (193, 173, 116) if (r + c) % 2 else (201, 183, 128)
            rect(surface, base, (x, y, CELL, CELL))
            rect(surface, (179, 158, 105), (x, y + 27, 28, 1))
            # Deterministic pixel texture, no external sprite assets.
            if (r * 7 + c * 3) % 4 == 0:
                rect(surface, (170, 157, 103), (x + 5, y + 8, 3, 2))
                rect(surface, (218, 199, 143), (x + 20, y + 21, 4, 2))
            if env.obstacle_map[r, c]:
                pygame.draw.polygon(surface, (93, 107, 98), [(x+3,y+23),(x+5,y+11),(x+13,y+5),(x+23,y+10),(x+25,y+24)])
                pygame.draw.polygon(surface, (135, 145, 122), [(x+5,y+11),(x+13,y+5),(x+18,y+12),(x+13,y+18)])
                rect(surface, (155, 163, 125), (x+8,y+8,5,2))
            if env.resource_map[r, c]:
                rect(surface, (115, 94, 59), (x+6,y+20,18,4))
                rect(surface, (108, 143, 108), (x+7,y+10,15,12))
                rect(surface, (194, 222, 152), (x+9,y+8,11,4))
                rect(surface, GOLD, (x+13,y+12,3,7))
                if (tick // 16 + r + c) % 4 == 0:
                    rect(surface, PAPER, (x+23,y+5,3,3))
            if env.panel_map[r, c]:
                rect(surface, (105, 95, 66), (x+3,y+23,23,3))
                rect(surface, INK, (x+7,y+20,3,5))
                rect(surface, INK, (x+20,y+20,3,5))
                rect(surface, (221, 226, 188), (x+3,y+6,23,16))
                rect(surface, (50, 93, 122), (x+5,y+8,19,12))
                for dx in (11, 18):
                    rect(surface, BLUE, (x+dx,y+8,1,12))
                rect(surface, BLUE, (x+5,y+14,19,1))
                if any(env.panel_map[p] for p in env.neighbors((r,c))):
                    rect(surface, GOLD, (x+23,y+4,3,3))
    for i, (r,c) in enumerate(env.robot_positions):
        draw_robot(surface, BX+c*CELL, BY+r*CELL, i, tick)


def draw(surface, env, controller, paused, editing, brush, tick, seed, speed):
    surface.fill(INK)
    for x,y in ((379,12),(411,28),(477,18),(463,42),(351,39)):
        rect(surface, (74, 94, 86), (x,y,2,2))
    text(surface, 'SOLAR GROVE', 16, 10, PAPER, 24)
    text(surface, 'A ROBOT EXPEDITION / BUILD A BRIGHTER OUTPOST', 17, 40, MUTED, 9)
    pygame.draw.circle(surface, GOLD, (483,28), 10)
    draw_world(surface, env, tick)
    text(surface, f'SECTOR {seed:05d}', 17, 297, MUTED, 10)
    status = 'MAP EDITOR' if editing else 'COMPLETE' if env.finished else 'PAUSED' if paused else 'EXPLORING'
    text(surface, status, 151, 297, GOLD, 10)
    panel(surface, (260, 60, 236, 71))
    text(surface, 'OUTPOST TELEMETRY', 272, 69, MUTED, 10)
    text(surface, f'{env.production():02d}', 272, 84, GOLD, 24)
    text(surface, 'POWER / TICK', 318, 88, PAPER, 10)
    text(surface, f'ENERGY {env.total_energy:,.0f}u  |  PANELS {env.panel_map.sum():02d}', 272, 115, PAPER, 10)
    panel(surface, (260, 140, 236, 78))
    for i in range(env.n_robots):
        y = 148+i*31
        text(surface, f'BOT {i+1:02d}', 272, y, BLUE if i == 0 else (232,145,111), 12)
        text(surface, ACTION_NAMES[env.last_actions[i]], 340, y+1, PAPER, 10)
        for slot in range(env.capacity):
            rect(surface, GREEN if env.inventory[i] > slot else (63,77,69), (452+slot*14,y+2,10,10))
        text(surface, 'MATERIAL', 430, y+15, MUTED, 8)
    panel(surface, (260, 227, 236, 65))
    text(surface, f'CONTROL / {controller.upper()}', 272, 236, GREEN, 11)
    text(surface, f'TICK {env.step_count:03d}/{env.max_steps}  RETURN {env.score:7.1f}', 272, 256, PAPER, 10)
    rect(surface, (20,36,39), (272,278,212,4))
    rect(surface, GREEN, (272,278,int(212*env.step_count/env.max_steps),4))
    if editing:
        footer = f'BRUSH: {brush.upper()} | 1 ROCK  2 SALVAGE  3 ERASE | CLICK TO PAINT'
    else:
        footer = 'SPACE PAUSE  R RESTART  N NEW MAP  E EDIT  TAB CONTROL  +/- SPEED'
    text(surface, footer, 16, 319, PAPER, 9)
    text(surface, f'{speed} TICKS/S  |  ABSTRACT ENERGY SIMULATION  |  ESC EXIT', 16, 336, MUTED, 9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=['greedy','random','ppo'], default='greedy')
    parser.add_argument('--model', default='models/solar_ppo.zip')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--screenshot', help='Save an offscreen screenshot and exit')
    parser.add_argument('--frames', type=int, default=80, help='Simulation ticks before screenshot')
    args = parser.parse_args()
    if args.screenshot:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    model = None
    if args.agent == 'ppo':
        if not Path(args.model).is_file():
            parser.error('PPO checkpoint missing. Run python train.py first, or use --agent greedy.')
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device='cpu')
    pygame.init()
    pygame.display.set_caption('Solar Grove | Multi-Robot PPO Expedition')
    screen = pygame.display.set_mode((W*2,H*2))
    canvas = pygame.Surface((W,H))
    env = MultiRobotSolarFarmEnv()
    seed = args.seed
    obs,_ = env.reset(seed=seed)
    controller = args.agent
    agents = {'greedy': GreedyAgent(env), 'random': RandomAgent(env.n_robots,seed)}
    if model is not None:
        agents['ppo'] = model
    paused, editing, brush, tick, elapsed, speed = False, False, 'rock', 0, 0.0, 6
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60)/1000 if not args.screenshot else 1/60
        tick += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and not editing:
                    paused = not paused
                elif event.key in (pygame.K_r, pygame.K_n):
                    seed += int(event.key == pygame.K_n)
                    obs,_ = env.reset(seed=seed)
                    agents['random'] = RandomAgent(env.n_robots,seed)
                    editing, paused, elapsed = False, False, 0
                elif event.key == pygame.K_e:
                    if not editing:
                        obs,_ = env.reset(seed=seed)
                    editing = not editing
                    paused = editing
                    obs = env._get_observation()
                    elapsed = 0
                elif event.key == pygame.K_TAB:
                    names = list(agents)
                    controller = names[(names.index(controller)+1)%len(names)]
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    speed = min(30,speed+2)
                elif event.key == pygame.K_MINUS:
                    speed = max(1,speed-2)
                elif event.key in (pygame.K_1,pygame.K_2,pygame.K_3):
                    brush = {pygame.K_1:'rock',pygame.K_2:'salvage',pygame.K_3:'erase'}[event.key]
            elif event.type == pygame.MOUSEBUTTONDOWN and editing and event.button == 1:
                mx,my = event.pos[0]//2, event.pos[1]//2
                r,c = (my-BY)//CELL, (mx-BX)//CELL
                if 0 <= r < 8 and 0 <= c < 8 and (r,c) not in env.robot_positions:
                    env.obstacle_map[r,c] = int(brush == 'rock')
                    env.resource_map[r,c] = int(brush == 'salvage')
                    env.panel_map[r,c] = 0
                    obs = env._get_observation()
        if not paused and not env.finished:
            elapsed += dt
            if args.screenshot or elapsed >= 1/speed:
                action,_ = agents[controller].predict(obs,deterministic=True)
                obs,_,_,_,_ = env.step(action)
                elapsed = 0
        draw(canvas,env,controller,paused,editing,brush,tick,seed,speed)
        pygame.transform.scale(canvas,screen.get_size(),screen)
        pygame.display.flip()
        if args.screenshot and tick >= args.frames:
            Path(args.screenshot).parent.mkdir(parents=True,exist_ok=True)
            pygame.image.save(screen,args.screenshot)
            running = False
    env.close()
    pygame.quit()


if __name__ == '__main__':
    import sys
    # Keep the original RL viewer available and its CLI compatible.
    if '--classic' in sys.argv:
        sys.argv.remove('--classic')
        main()
    elif any(flag in sys.argv for flag in ('--agent', '--model', '--seed')):
        main()
    else:
        from solar_target import main as target_main
        target_main()
