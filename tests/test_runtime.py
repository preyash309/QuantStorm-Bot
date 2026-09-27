"""Regressions for the recommended runtime, loader and CLI."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import play_match
from game_config import GameConfig, ConfigError
from bot_loader import load_for_testing
from strategies.adaptive_standalone import Bot
from strategies.rational import Bot as Rational

class RuntimeTests(unittest.TestCase):
    def test_reproducible_zero_sum(self):
        config = GameConfig()
        a = play_match(Bot, Rational, config, seed=42, mirror=True, n_deals=3, verbose=False)
        b = play_match(Bot, Rational, config, seed=42, mirror=True, n_deals=3, verbose=False)
        self.assertEqual(a.pnl, b.pnl)
        self.assertAlmostEqual(sum(a.pnl), 0)
        self.assertEqual(len(a.deals), 6)
        self.assertFalse(any(a.forfeits))
        self.assertFalse(a.bot_a_warnings)

    def test_loader_restored(self):
        loaded = load_for_testing(str(ROOT / 'strategies/adaptive_standalone.py'))
        self.assertEqual(loaded.name, Bot.name)

    def test_default_paths_from_other_directory(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'backtester.py'),
                                 '--n_deals', '1', '--seed', '42', '--quiet'],
                                cwd=ROOT/'tests', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn('MATCH SUMMARY', result.stdout)

    def test_invalid_deals(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT/'backtester.py'),
                                 '--n_deals', '0'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must be positive', result.stderr)

    def test_frozen_rules(self):
        config = GameConfig()
        with self.assertRaises(ConfigError):
            config.TE_BUDGET = 100

if __name__ == '__main__':
    unittest.main()
