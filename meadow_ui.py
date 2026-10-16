"""Original procedural pixel art for Solar Grove's handheld-style meadow UI."""
import argparse
import math
import os
from pathlib import Path
import random
import textwrap

from solar_target import Farm, REGIONS, COLORS, DIRECTIONS, sun_vector

WIDTH, HEIGHT = 640, 450
ORIGIN, TILE = (210, 112), 22
INK = (38, 57, 54)
CREAM = (251, 244, 210)
GOLD = (242, 191, 85)
GRASS = (139, 182, 100)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screenshot')
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--designer', choices=['ppo', 'planner'])
    parser.add_argument('--array-model', default='checkpoints/array_ppo.zip')
    args = parser.parse_args()
    if args.screenshot:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    import pygame as pg
    pg.init()
    display = pg.display.set_mode((1280, 900))
    pg.display.set_caption('Solar Grove | Meadow Mission')
    canvas = pg.Surface((WIDTH, HEIGHT))
    fonts = {s: pg.font.Font(pg.font.match_font('menlo,monospace'), s) for s in (8, 9, 10, 12, 16)}

    def rect(color, x, y, w, h, border=0):
        pg.draw.rect(canvas, color, (int(x), int(y), int(w), int(h)), border)

    def text(value, x, y, size=9, color=INK):
        canvas.blit(fonts[size].render(str(value), False, color), (int(x), int(y)))

    def dialog(x, y, w, h):
        rect((54, 79, 67), x+3, y+4, w, h)
        rect(INK, x, y, w, h)
        rect(CREAM, x+2, y+2, w-4, h-4)
        rect((179, 194, 137), x+5, y+5, w-10, h-10, 1)

    def point(pos):
        return (ORIGIN[0]+pos[0]*TILE+TILE//2, ORIGIN[1]+pos[1]*TILE+TILE//2)

    def pine(x, y, scale=1):
        x, y = int(x), int(y)
        rect((100, 93, 63), x-2*scale, y-7*scale, 4*scale, 11*scale)
        for offset, spread, shade in ((0, 13, (45, 100, 74)), (-9, 11, (58, 120, 79)), (-18, 8, (74, 138, 86))):
            pg.draw.polygon(canvas, shade, [(x, y+(offset-21)*scale), (x-spread*scale, y+offset*scale), (x+spread*scale, y+offset*scale)])
            rect((96, 158, 95), x-2*scale, y+(offset-12)*scale, 3*scale, 3*scale)

    def mountain(points, color):
        pg.draw.polygon(canvas, color, points)

    rng = random.Random(42)
    tufts = [(rng.randrange(640), rng.randrange(144, 364), rng.randrange(5)) for _ in range(670)]
    designer = None
    controller = args.designer or ('ppo' if Path(args.array_model).is_file() else 'planner')
    if controller == 'ppo':
        if not Path(args.array_model).is_file():
            parser.error('Array PPO checkpoint missing. Run train_array.py or use --designer planner.')
        from array_rl import LearnedDesigner
        designer = LearnedDesigner(args.array_model)
    farm = Farm(designer=designer)
    farm.command('produce 1000 W')
    command_text, entering = 'produce 1000 W', False
    paused, daylight = False, False
    clock = pg.time.Clock()
    elapsed, frames, time = 0.0, 0, 0.0
    visual_positions = [list(map(float, point(p))) for p in farm.robots]
    running = True

    def background(t):
        canvas.fill((183, 220, 202))
        # Layered granite silhouettes and a broad dome, drawn as original pixels.
        mountain([(0, 135), (0, 88), (38, 60), (66, 82), (105, 52), (148, 96), (187, 76), (241, 128)], (139, 171, 158))
        mountain([(360, 133), (395, 71), (427, 94), (467, 50), (508, 81), (542, 61), (589, 99), (640, 69), (640, 150)], (143, 171, 157))
        mountain([(102, 132), (124, 111), (142, 63), (149, 58), (187, 58), (195, 71), (202, 126)], (178, 188, 166))
        mountain([(115, 130), (138, 80), (149, 66), (153, 116), (170, 130)], (154, 169, 151))
        rect((213, 214, 187), 150, 59, 35, 5)
        for x in (157, 170, 185):
            rect((155, 173, 153), x, 76, 2, 39)
        pg.draw.ellipse(canvas, (193, 202, 179), (278, 68, 87, 99))
        rect((170, 184, 164), 321, 90, 44, 65)
        mountain([(278, 154), (304, 95), (320, 83), (321, 155)], (210, 214, 186))
        rect((170, 193, 154), 0, 128, 640, 35)
        pg.draw.ellipse(canvas, (150, 185, 119), (-80, 125, 400, 100))
        pg.draw.ellipse(canvas, (151, 187, 119), (354, 130, 370, 100))
        rect(GRASS, 0, 160, 640, 207)
        # Small creek follows the western edge, away from the construction plot.
        river = [(39, 151), (51, 188), (43, 228), (67, 270), (57, 312), (31, 367)]
        pg.draw.lines(canvas, (106, 153, 110), False, river, 25)
        pg.draw.lines(canvas, (111, 184, 191), False, river, 17)
        for x, y in ((43, 190), (48, 232), (61, 282), (48, 332)):
            rect((177, 222, 211), x+int(math.sin(t*2+y)*2), y, 8, 2)
        # Dirt trail leading to the solar farm's service gate.
        pg.draw.lines(canvas, (182, 173, 119), False, [(130, 367), (175, 339), (254, 339), (254, 310), (267, 310)], 13)
        pg.draw.lines(canvas, (203, 193, 139), False, [(130, 367), (175, 339), (254, 339), (254, 310), (267, 310)], 8)
        for x, y, variant in tufts:
            if 263 < x < 370 and 165 < y < 300:
                continue
            if x < 80:
                continue
            if variant == 0:
                rect((111, 159, 87), x, y, 1, 3)
                rect((111, 159, 87), x+2, y+1, 1, 2)
            elif variant == 1:
                rect((174, 203, 124), x, y, 3, 1)
            elif variant == 2 and (x+y)%4 == 0:
                rect((240, 227, 154), x, y, 2, 2)
                rect((240, 244, 209), x+1, y-1, 1, 1)
        for x, y, scale in ((12, 174, 1), (88, 160, 1), (111, 192, 1), (202, 164, 1), (440, 165, 1), (470, 160, 1), (615, 156, 1), (598, 345, 1), (411, 349, 1), (13, 349, 1), (89, 363, 1)):
            pine(x, y, scale)

    def animal(kind, x, y, t, facing=1):
        # Wildlife wanders only outside the farm footprint.
        x, y = int(x), int(y)
        bob = int(math.sin(t*5+x) > 0)
        pg.draw.ellipse(canvas, (116, 158, 91), (x-2, y+9, 24, 5))
        if kind == 'deer':
            color = (171, 128, 77)
            rect(color, x, y, 17, 9)
            rect((202, 161, 103), x+2, y, 11, 3)
            hx = x+15 if facing > 0 else x-4
            rect(color, hx, y-6, 6, 12)
            rect(color, hx+1, y-9, 7, 5)
            rect((224, 196, 135), hx, y-13, 2, 5)
            rect((224, 196, 135), hx+5, y-12, 2, 5)
            rect(INK, hx+5 if facing > 0 else hx+1, y-7, 1, 1)
            for offset in (2, 6, 12, 15):
                rect((112, 92, 61), x+offset, y+7, 2, 7-bob)
        elif kind == 'rabbit':
            rect((230, 221, 188), x+3, y+2, 10, 8)
            rect((241, 232, 202), x+11, y, 7, 7)
            rect((241, 232, 202), x+12, y-7-bob, 2, 8)
            rect((241, 232, 202), x+16, y-6-bob, 2, 8)
            rect((255, 246, 222), x, y+4, 4, 4)
            rect(INK, x+16, y+2, 1, 1)
        else:
            rect((236, 229, 193), x+1, y, 17, 11)
            rect((249, 243, 215), x+3, y-2, 12, 12)
            rect((112, 108, 82), x+16, y+2, 6, 7)
            for offset in (4, 13):
                rect((112, 108, 82), x+offset, y+10, 2, 5-bob)

    def world(t):
        east, south, up = sun_vector(REGIONS[farm.region][1], farm.hour)
        # Tree silhouettes cast directional ground shadows without a visible grid.
        for pos, height in farm.obstacles.items():
            x, y = point(pos)
            length = min(90, height*16/max(.15, up))
            pg.draw.line(canvas, (107, 148, 91), (x, y+8), (int(x-east*length), int(y+8-south*length)), 14)
        # Prepared central plot and an access aisle between every rack.
        rect((122, 160, 92), 268, 171, 96, 117)
        for row in (3, 5, 7):
            x, y = point((3, row))
            rect((177, 176, 119), x-10, y-11, 89, 23)
            rect((153, 163, 108), x-10, y+12, 89, 5)
        # Low perimeter markers: open corridors on all four sides.
        for x in (263, 370):
            for y in (166, 205, 249, 294):
                rect((98, 117, 78), x, y, 2, 7)
                rect((223, 218, 164), x, y, 2, 2)
        # Short series links under racks, then orthogonal cable trenches to box.
        for g in range(3):
            row = sorted([pos for pos, p in farm.panels.items() if p.group == g])
            if not row:
                continue
            x0, yy = point(row[0]); x1, _ = point(row[-1])
            color = (159, 137, 88)
            pg.draw.line(canvas, color, (x0-5, yy+10), (x1+9, yy+10), 2)
            lane = 369+g*3
            pg.draw.lines(canvas, color, False, [(x1+9, yy+10), (lane, yy+10), (lane, 306), (353, 306)], 2)
        # Combiner / inverter cabinet at service entrance.
        rect((80, 100, 82), 343, 297, 18, 20)
        rect((220, 220, 179), 343, 294, 17, 19)
        rect((68, 123, 114), 346, 298, 10, 5)
        rect(GOLD, 349, 306, 3, 4)
        text('COMBINER', 330, 322, 8, INK)
        for pos, p in farm.panels.items():
            x, y = point(pos)
            if not p.built:
                rect((129, 147, 102), x-10, y-8, 20, 16, 1)
                for dx in (-8, 7):
                    rect((214, 203, 146), x+dx, y-6, 2, 14)
                rect((200, 191, 130), x-8, y+3, 17, 2)
                continue
            # Tilted rack silhouette, feet and individual cell glints.
            rect((97, 113, 84), x-10, y+7, 21, 4)
            for dx in (-7, 7):
                rect((119, 122, 94), x+dx, y+2, 2, 8)
            pg.draw.polygon(canvas, (229, 230, 199), [(x-10,y-8),(x+9,y-8),(x+11,y+6),(x-8,y+6)])
            pg.draw.polygon(canvas, (42, 82, 114), [(x-8,y-6),(x+7,y-6),(x+9,y+4),(x-6,y+4)])
            for dx in (-3, 3):
                pg.draw.line(canvas, (112, 164, 187), (x+dx,y-6), (x+dx+2,y+4))
            pg.draw.line(canvas, (112, 164, 187), (x-7,y-1), (x+8,y-1))
            rect((174, 207, 205), x-7, y-5, 4, 1)
            if farm.shadowed(pos):
                shade = pg.Surface((20, 15), pg.SRCALPHA)
                shade.fill((24, 47, 53, 115))
                canvas.blit(shade, (x-10, y-8))
        for pos in sorted(farm.obstacles, key=lambda p: p[1]):
            x, y = point(pos)
            pine(x, y+8)
        for i, pos in enumerate(farm.robots):
            tx, ty = point(pos)
            for axis, target in enumerate((tx, ty)):
                visual_positions[i][axis] += (target-visual_positions[i][axis])*.28
            if args.screenshot:
                visual_positions[i][:] = [tx, ty]
            x, y = map(int, visual_positions[i])
            pg.draw.ellipse(canvas, (102, 139, 88), (x-10,y+8,22,5))
            for offset in (-10, 6):
                rect((43, 64, 59), x+offset, y-5, 5, 15)
                rect((91, 112, 89), x+offset+1, y-3, 2, 3)
                rect((91, 112, 89), x+offset+1, y+4, 2, 3)
            rect((245, 222, 155), x-6, y-10, 13, 17)
            rect((255, 244, 201), x-5, y-10, 11, 3)
            rect((37, 84, 86), x-4, y-5, 9, 5)
            rect((133, 218, 195), x-3, y-4, 2, 2)
            rect((133, 218, 195), x+2, y-4, 2, 2)
            rect((101, 115, 91), x, y-16, 1, 6)
            rect(COLORS[i], x-1, y-17, 3, 3)
            if farm.jobs[i] == pos:
                for dx, dy in ((11, -3), (14, -6), (10, -9)):
                    if int(t*8+dx)%3:
                        rect(GOLD, x+dx, y+dy, 2, 2)
        animal('deer', 130+math.sin(t*.2)*22, 243+math.sin(t*.3)*6, t)
        animal('deer', 444+math.sin(t*.15)*14, 293+math.sin(t*.25)*7, t, -1)
        animal('rabbit', 187+math.sin(t*.24)*12, 292, t)
        animal('sheep', 396+math.sin(t*.18)*10, 212, t)
        animal('rabbit', 105+math.sin(t*.2)*8, 327, t+2)

    while running:
        dt = 1/60 if args.screenshot else clock.tick(60)/1000
        frames += 1
        time += dt
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                x, y = event.pos[0]//2, event.pos[1]//2
                if 14 <= x <= 626 and 370 <= y <= 436:
                    entering = True
                    command_text = ''
            elif event.type == pg.KEYDOWN:
                key = event.key
                if entering:
                    if key == pg.K_RETURN:
                        if farm.command(command_text):
                            entering = False
                            paused = False
                    elif key == pg.K_ESCAPE:
                        entering = False
                    elif key == pg.K_BACKSPACE:
                        command_text = command_text[:-1]
                    elif len(command_text) < 35:
                        command_text += getattr(event, 'unicode', '')
                    continue
                if key == pg.K_ESCAPE:
                    running = False
                elif key == pg.K_RETURN:
                    entering, command_text = True, ''
                elif key == pg.K_SPACE:
                    paused = not paused
                elif key == pg.K_l:
                    farm.region = (farm.region+1)%len(REGIONS)
                    farm.best = 0
                    farm.plan()
                elif key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                    farm.hour = max(6, min(18, farm.hour+(-.5 if key == pg.K_LEFTBRACKET else .5)))
                    farm.plan()
                elif key == pg.K_d:
                    daylight = not daylight
                elif key == pg.K_r:
                    farm = Farm(designer=designer)
                    farm.command('produce 1000 W')
                    visual_positions = [list(map(float, point(p))) for p in farm.robots]
                    paused, daylight = False, False
        if not paused and not entering:
            elapsed += dt
            if elapsed >= .22 or args.screenshot:
                farm.step()
                elapsed = 0
            if daylight:
                old_slot = int(farm.hour*2)
                farm.hour = min(18, farm.hour+dt/20)
                if int(farm.hour*2) != old_slot:
                    farm.plan()
                if farm.hour == 18:
                    daylight = False
        background(time)
        world(time)
        power = farm.power()
        # RPG status plaque, clear goals and a single primary command action.
        dialog(12, 10, 225, 43)
        text('SOLAR GROVE', 24, 17, 16)
        text('MEADOW MISSION / ROBOT FIELD TEAM', 25, 38, 8, (94, 119, 80))
        dialog(405, 10, 222, 65)
        text('PRODUCTION', 417, 20, 8)
        text(f'{power:,.0f} / {farm.target:,.0f} W', 417, 34, 12)
        rect((203, 211, 165), 418, 56, 193, 8)
        rect((83, 151, 105), 418, 56, int(193*min(1, power/farm.target)), 8)
        dialog(474, 131, 152, 152)
        text('PPO FIELD TEAM' if designer else 'PLANNER FIELD TEAM', 486, 143, 9)
        text(REGIONS[farm.region][0].upper(), 486, 164, 10)
        text(f'SUN {int(farm.hour):02d}:{int(farm.hour%1*60):02d}', 486, 181, 9)
        east, south, up = sun_vector(REGIONS[farm.region][1], farm.hour)
        text(f'HEIGHT {math.degrees(math.asin(max(-1,min(1,up)))):.0f} deg', 486, 195, 8)
        built = sum(p.built for p in farm.panels.values())
        text(f'BUILT {built}/{len(farm.panels)}', 486, 215, 9)
        text('4 MODULES / STRING', 486, 234, 8)
        text(f'{len(farm.panels)//4} STRINGS IN PARALLEL', 486, 247, 8)
        text('L PLACE  [ ] TIME', 486, 266, 8, (94, 119, 80))
        # A small engineering card keeps the meadow the main focus.
        dialog(14, 93, 150, 66)
        text('ARRAY / 4S x '+str(len(farm.panels)//4)+'P', 25, 103, 9)
        text('80 V RATED / EQUAL STRINGS', 25, 120, 8)
        first = next(iter(farm.panels.values()), None)
        if first:
            text(f'FACE {DIRECTIONS[first.facing]} / TILT {first.tilt} deg', 25, 137, 8)
        status = farm.mission_status()
        dialog(14, 371, 612, 65)
        rect((218, 225, 175), 24, 382, 36, 41)
        text('R1', 32, 395, 12)
        if entering:
            text('YOUR PRODUCTION ORDER', 72, 380, 9, (94, 119, 80))
            text('> '+command_text+('_' if frames//20%2 == 0 else ''), 72, 396, 12)
            text('ENTER SEND   ESC CANCEL   e.g. produce 1 kW', 72, 416, 8)
        else:
            heading = 'MISSION COMPLETE!' if power >= farm.target else 'DESIGN BELOW TARGET' if 'BELOW TARGET' in status else 'ROBOT TEAM / '+('PAUSED' if paused else 'BUILDING')
            text(heading, 72, 380, 10)
            lines = textwrap.wrap(farm.notice, width=80)
            for i, line in enumerate(lines[:2]):
                text(line, 72, 395+i*11, 8)
            text('ENTER NEW ORDER   SPACE PAUSE   D SUN WALK   R RESTART', 72, 420, 8, (94, 119, 80))
        text(('LEARNED PPO DESIGN + BFS ROBOTS' if designer else 'PLANNER DESIGN + BFS ROBOTS')+' / EDUCATIONAL PV MODEL', 15, 440, 8, CREAM)
        pg.transform.scale(canvas, display.get_size(), display)
        pg.display.flip()
        if args.screenshot and frames >= args.frames:
            Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
            pg.image.save(display, args.screenshot)
            running = False
    pg.quit()
