"""Solar Grove production puzzle. Deliberately simplified educational physics."""
from collections import deque
from dataclasses import dataclass
import math
import re

REGIONS = [('California', 37), ('Kenya', 0), ('Australia', -34), ('Norway', 60)]
COLORS = [(111, 200, 225), (249, 180, 93), (193, 149, 233)]
DIRECTIONS = ['N', 'E', 'S', 'W']
VECTORS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


@dataclass
class Panel:
    group: int
    facing: int
    tilt: int = 30
    built: bool = False


def sun_vector(latitude, hour):
    """Unit vector east, south, up at equinox, using local solar time."""
    lat, angle = math.radians(latitude), math.radians(15 * (hour - 12))
    return (-math.sin(angle), math.sin(lat) * math.cos(angle),
            math.cos(lat) * math.cos(angle))


def string_output(exposures):
    """Ideal 20 V, 5 A panels, no bypass diodes or IV curve model."""
    if not exposures:
        return 0.0, 0.0, 0.0
    voltage, current = 20 * len(exposures), 5 * min(exposures)
    return voltage, current, voltage * current


class Farm:
    size = 10
    budget = 12
    target = 1000

    def __init__(self, designer=None):
        self.designer = designer
        self.region = 0
        self.hour = 12.0
        self.panels = {}
        self.obstacles = {(1, 2): 2.5, (8, 4): 3.0, (1, 7): 2.0, (8, 8): 2.5}
        self.robots = [(2, 8), (7, 8)]
        self.jobs = [None, None]
        self.progress = [0, 0]
        self.notice = 'Press Enter to command a production target.'
        self.autonomous = False
        self.predicted = 0.0
        self.best = 0.0

    def place(self, pos, group, facing, tilt):
        if pos in self.obstacles or pos in self.robots:
            self.notice = 'Choose an empty tile away from a robot.'
            return
        if pos in self.panels:
            p = self.panels[pos]
            p.group, p.facing, p.tilt = group, facing, tilt
            self.notice = 'Panel settings updated.'
        elif len(self.panels) < self.budget:
            self.panels[pos] = Panel(group, facing, tilt)
            self.notice = 'Build order added.'
        else:
            self.notice = '12-panel limit. Right-click to reclaim one.'

    def remove(self, pos):
        self.panels.pop(pos, None)
        self.notice = 'Panel reclaimed; material is reusable.'

    def shadowed(self, pos):
        east, south, up = sun_vector(REGIONS[self.region][1], self.hour)
        if up <= 0:
            return True
        # Cast a ray toward the sun; each tile is one abstract distance unit.
        for step in range(1, 301):
            distance = step * 0.1
            x, y = pos[0] + east * distance, pos[1] + south * distance
            tile = (math.floor(x + .5), math.floor(y + .5))
            if not (0 <= tile[0] < self.size and 0 <= tile[1] < self.size):
                break
            if self.obstacles.get(tile, 0) > up * distance:
                return True
        return False

    def exposure(self, pos):
        p = self.panels[pos]
        return self.panel_exposure(pos, p.facing, p.tilt)

    def panel_exposure(self, pos, facing, tilt):
        if self.shadowed(pos):
            return 0.0
        east, south, up = sun_vector(REGIONS[self.region][1], self.hour)
        dx, dy = VECTORS[facing]
        angle = math.radians(tilt)
        incidence = dx * math.sin(angle) * east + dy * math.sin(angle) * south + math.cos(angle) * up
        return max(0, incidence) * max(0, up) ** .15

    def command(self, command):
        match = re.fullmatch(r'(?:produce\s+)?([0-9]+(?:\.[0-9]+)?)\s*(kw|w|watts)?', command.strip().lower())
        if not match:
            self.notice = 'Try: produce 1000 W (or produce 1 kW).'
            return False
        target = float(match[1]) * (1000 if match[2] == 'kw' else 1)
        if not 0 < target <= 1000000:
            self.notice = 'Choose a target from 1 to 1,000,000 watts.'
            return False
        self.target = target
        self.autonomous = True
        self.plan()
        return True

    def plan(self):
        if self.designer is not None:
            self.designer.apply(self)
            return
        # Search finite panel settings, rank reachable unshaded sites, and group
        # similar exposure into strings to avoid dragging bright panels down.
        east, south, up = sun_vector(REGIONS[self.region][1], self.hour)
        def incidence(f, t):
            dx, dy = VECTORS[f]
            a = math.radians(t)
            return max(0, dx*math.sin(a)*east + dy*math.sin(a)*south + math.cos(a)*up)
        _, facing, tilt = max((incidence(f, t), f, t) for f in range(4) for t in range(0, 61, 15))
        # A planned rack has four modules in series. Parallel strings have
        # identical module counts, with one empty tile between rack rows.
        rows = [[(x, y) for x in range(3, 7)] for y in (3, 5, 7)]
        import itertools
        designs = []
        for count in range(1, 4):
            for chosen in itertools.combinations(rows, count):
                if any(pos in self.obstacles for row in chosen for pos in row):
                    continue
                if any(not any(self.path(bot, pos, set(self.obstacles)) is not None
                               for bot in self.robots) for row in chosen for pos in row):
                    continue
                watts = sum(string_output([self.panel_exposure(pos, facing, tilt)
                                           for pos in row])[2] for row in chosen)
                designs.append((watts, chosen))
            feasible = [design for design in designs if design[0] >= self.target]
            if feasible:
                break
        if feasible:
            predicted, chosen = max(feasible, key=lambda design: design[0])
        elif designs:
            predicted, chosen = max(designs, key=lambda design: design[0])
        else:
            predicted, chosen = 0.0, []
        previous = self.panels
        self.panels = {}
        for group, row in enumerate(chosen):
            for pos in row:
                old = previous.get(pos)
                built = bool(old and old.built and (old.group, old.facing, old.tilt) == (group, facing, tilt))
                self.panels[pos] = Panel(group, facing, tilt, built)
        self.jobs, self.progress = [None, None], [0, 0]
        self.predicted = predicted
        self.notice = (f'Building {len(chosen)} rows of 4. Expected {predicted:.0f} W. '
                       f'Facing {DIRECTIONS[facing]}, tilt {tilt} degrees.')
        if predicted < self.target:
            self.notice = f'Target exceeds sunlight capacity. Building best plan: {predicted:.0f} W.'

    def mission_status(self):
        if self.power() >= self.target:
            return 'TARGET REACHED'
        if self.autonomous and all(p.built for p in self.panels.values()):
            return 'DESIGN BELOW TARGET / CHANGE SUN OR TARGET'
        return 'ROBOTS BUILDING' if self.autonomous else 'AWAITING COMMAND'

    def outputs(self):
        groups = []
        for group in range(3):
            exposures = [self.exposure(pos) for pos, p in self.panels.items()
                         if p.built and p.group == group]
            if self.autonomous and len(exposures) != 4:
                groups.append((0.0, 0.0, 0.0))
            else:
                groups.append(string_output(exposures))
        return groups

    def power(self):
        return sum(row[2] for row in self.outputs())

    def path(self, start, goal, blocked):
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            pos, path = queue.popleft()
            if pos == goal:
                return path
            for dx, dy in VECTORS:
                nxt = (pos[0] + dx, pos[1] + dy)
                if (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size
                        and nxt not in blocked and nxt not in seen):
                    seen.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

    def step(self):
        for i in range(2):
            job = self.jobs[i]
            if job not in self.panels or self.panels[job].built:
                self.jobs[i] = None
                self.progress[i] = 0
            if self.jobs[i] is None:
                choices = []
                blocked = set(self.obstacles) | {self.robots[1-i]}
                for pos, p in self.panels.items():
                    if not p.built and pos not in self.jobs:
                        path = self.path(self.robots[i], pos, blocked)
                        if path is not None:
                            choices.append((len(path), pos))
                if choices:
                    self.jobs[i] = min(choices)[1]
            job = self.jobs[i]
            if job is None:
                parking = ((2, 8), (7, 8))[i]
                path = self.path(self.robots[i], parking, set(self.obstacles) | {self.robots[1-i]})
                if path:
                    self.robots[i] = path[0]
                continue
            if self.robots[i] == job:
                self.progress[i] += 1
                if self.progress[i] >= 3:
                    self.panels[job].built = True
                    self.jobs[i] = None
                    self.progress[i] = 0
            else:
                path = self.path(self.robots[i], job, set(self.obstacles) | {self.robots[1-i]})
                if path:
                    self.robots[i] = path[0]
        self.best = max(self.best, self.power())


def main():
    from meadow_ui import run
    run()


if __name__ == '__main__':
    main()
