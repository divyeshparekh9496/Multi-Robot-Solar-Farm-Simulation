import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pygame

import game


class GameTests(unittest.TestCase):
    def test_controls_editor_and_screenshot(self):
        key = lambda code: pygame.event.Event(pygame.KEYDOWN, key=code)
        click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(50, 150))
        batches = [[key(pygame.K_e)], [key(pygame.K_2), click],
                   [key(pygame.K_1), click], [key(pygame.K_3), click],
                   [key(pygame.K_e)], [key(pygame.K_SPACE)],
                   [key(pygame.K_TAB)], [key(pygame.K_r)],
                   [key(pygame.K_n)], [key(pygame.K_EQUALS), key(pygame.K_MINUS)]]
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / 'preview.png')
            with patch('sys.argv', ['game.py','--screenshot',output,'--frames','10']), \
                    patch('pygame.event.get', side_effect=batches):
                game.main()
            self.assertGreater(Path(output).stat().st_size, 1000)

    def test_missing_ppo_checkpoint_exits_clearly(self):
        with patch('sys.argv', ['game.py','--agent','ppo','--model','missing.zip']):
            with self.assertRaises(SystemExit) as raised:
                game.main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
