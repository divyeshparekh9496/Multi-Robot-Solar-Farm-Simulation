import unittest
from solar_target import Farm, Panel, sun_vector, string_output


class SolarTargetTests(unittest.TestCase):
    def test_hemisphere_reverses_noon_direction(self):
        self.assertGreater(sun_vector(37, 12)[1], 0)
        self.assertLess(sun_vector(-34, 12)[1], 0)
        self.assertAlmostEqual(sum(v*v for v in sun_vector(37, 9)), 1)

    def test_series_shade_limits_current_parallel_channels_stay_independent(self):
        self.assertEqual(string_output([1, .2]), (40, 1, 40))
        self.assertEqual(string_output([1, 1])[2] + string_output([.2, .2])[2], 240)
        self.assertEqual(string_output([]), (0, 0, 0))

    def test_tree_casts_shadow_and_direction_changes_exposure(self):
        farm = Farm()
        self.assertTrue(farm.shadowed((1, 1)))
        farm.panels[(0, 4)] = Panel(0, 2, built=True)
        south = farm.exposure((0, 4))
        farm.panels[(0, 4)].facing = 0
        self.assertGreater(south, farm.exposure((0, 4)))
        farm.hour = 0
        self.assertEqual(farm.exposure((0, 4)), 0)

    def test_robots_build_reachable_design_and_target_can_be_met(self):
        farm = Farm()
        sites = [(x, y) for y in range(5, 10) for x in range(10)
                 if (x, y) not in farm.obstacles and not farm.shadowed((x, y))]
        for j, pos in enumerate(sites[:12]):
            farm.place(pos, j//4, 2, 30)
        for _ in range(300):
            farm.step()
            self.assertEqual(len(set(farm.robots)), 2)
            self.assertFalse(set(farm.robots) & set(farm.obstacles))
        self.assertEqual(sum(p.built for p in farm.panels.values()), 12)
        self.assertGreater(farm.power(), farm.target)

    def test_budget_and_reclaim_cancel_pending_jobs(self):
        farm = Farm()
        for x in range(10):
            farm.place((x, 5), 0, 2, 30)
        farm.place((0, 6), 0, 2, 30)
        farm.place((1, 6), 0, 2, 30)
        farm.place((2, 6), 0, 2, 30)
        self.assertEqual(len(farm.panels), 12)
        farm.step()
        for pos in list(farm.panels):
            farm.remove(pos)
        farm.step()
        self.assertEqual(farm.jobs, [None, None])

    def test_interactive_controls_and_render(self):
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        import pygame
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import solar_target
        def key(value):
            return pygame.event.Event(pygame.KEYDOWN, key=value)
        click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(55, 430))
        events = [[key(pygame.K_2), key(pygame.K_f), key(pygame.K_t), click],
                  [key(pygame.K_l), key(pygame.K_LEFTBRACKET)],
                  [key(pygame.K_d), key(pygame.K_SPACE)], [key(pygame.K_r)]]
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory)/'target.png')
            with patch('sys.argv', ['solar_target.py', '--screenshot', output, '--frames', '4']), patch('pygame.event.get', side_effect=events):
                solar_target.main()
            self.assertGreater(Path(output).stat().st_size, 1000)

    def test_production_command_builds_target_without_manual_placement(self):
        farm = Farm()
        self.assertTrue(farm.command('produce 1 kW'))
        self.assertEqual(farm.target, 1000)
        self.assertTrue(farm.autonomous)
        self.assertLessEqual(len(farm.panels), farm.budget)
        for _ in range(200):
            farm.step()
        self.assertGreaterEqual(farm.power(), farm.target)
        self.assertEqual(farm.mission_status(), 'TARGET REACHED')
        self.assertAlmostEqual(farm.power(), farm.predicted)

    def test_unachievable_target_and_bad_command_are_explicit(self):
        farm = Farm()
        self.assertTrue(farm.command('produce 2000 W'))
        self.assertLess(farm.predicted, farm.target)
        for _ in range(200):
            farm.step()
        self.assertIn('BELOW TARGET', farm.mission_status())
        panels = dict(farm.panels)
        for command in ('hello', 'produce 0 W', 'produce -1', 'inf', '1e999'):
            self.assertFalse(farm.command(command))
            self.assertEqual(farm.panels, panels)
            self.assertEqual(farm.target, 2000)

    def test_autonomy_in_both_hemispheres_and_low_sun(self):
        for region in range(4):
            for hour in (7, 12, 17):
                farm = Farm()
                farm.region, farm.hour = region, hour
                farm.command('produce 1000 W')
                for _ in range(200):
                    farm.step()
                    self.assertEqual(len(set(farm.robots)), 2)
                self.assertTrue(all(p.built for p in farm.panels.values()))
                self.assertAlmostEqual(farm.power(), farm.predicted)

    def test_array_rows_have_equal_strings_service_aisles_and_no_partial_connection(self):
        farm = Farm()
        farm.command('produce 1000 W')
        self.assertEqual(farm.power(), 0)
        groups = [[pos for pos, p in farm.panels.items() if p.group == g] for g in range(3)]
        for group in groups:
            self.assertEqual(len(group), 4)
            self.assertEqual(sorted(x for x, _ in group), [3, 4, 5, 6])
            self.assertEqual(len(set(y for _, y in group)), 1)
        self.assertEqual(sorted({y for _, y in farm.panels}), [3, 5, 7])
        self.assertFalse(set(farm.panels) & set(farm.obstacles))
        self.assertEqual(len({(p.facing, p.tilt) for p in farm.panels.values()}), 1)
        for pos in groups[0][:3]:
            farm.panels[pos].built = True
        self.assertEqual(farm.power(), 0)
        farm.panels[groups[0][3]].built = True
        self.assertGreater(farm.power(), 0)
        self.assertEqual(farm.outputs()[0][0], 80)

    def test_command_entry_in_new_interface(self):
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        import pygame
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import meadow_ui
        farm = Farm()
        keys = [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)]
        keys += [pygame.event.Event(pygame.KEYDOWN, key=ord(c), unicode=c) for c in 'produce 600 W']
        keys.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory)/'command.png')
            with patch('sys.argv', ['game.py', '--screenshot', output, '--frames', '2']), patch('pygame.event.get', side_effect=[keys, []]), patch('meadow_ui.Farm', return_value=farm):
                meadow_ui.run()
            self.assertEqual(farm.target, 600)
            self.assertEqual(len(farm.panels), 8)
            self.assertGreater(Path(output).stat().st_size, 1000)
