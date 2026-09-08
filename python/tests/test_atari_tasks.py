from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from atari_tasks import EpisodeTask, QBERT_CUBES, ROM_SHA256


def task(game):
    environment = f'ALE/{game}-v5'
    return EpisodeTask(environment, ROM_SHA256[environment])


def pyramid(color):
    ram = [0] * 128
    for index in QBERT_CUBES:
        ram[index] = color
    return ram


def test_qbert_needs_all_cubes_not_a_large_score():
    observer = task('Qbert')
    observer.advance(0, pyramid(148))
    ram = pyramid(26)
    ram[QBERT_CUBES[-1]] = 148
    observer.advance(10000, ram)
    assert not observer.result(True, False)['episode_success']
    observer.advance(25, pyramid(26))
    assert observer.result(True, False)['episode_success']
    assert observer.first_milestone_frame == 3
    assert observer.max_initial_cubes == 21
    observer.advance(100, pyramid(231))
    observer.advance(0, pyramid(26))
    assert observer.first_milestone_frame == 3


def test_qbert_reset_and_uniform_initial_colors_are_not_completion():
    observer = task('Qbert')
    observer.advance(0, pyramid(0))
    observer.advance(0, pyramid(26))
    assert not observer.result(True, False)['episode_success']
    observer.advance(0, pyramid(148))
    assert not observer.result(True, False)['episode_success']
    observer.advance(25, pyramid(26))
    assert observer.result(True, False)['episode_success']
    assert not task('Qbert').result(False, False)['episode_success']


def test_reverted_cubes_must_be_recovered_not_counted_as_cumulative_visits():
    observer = task('Qbert')
    observer.advance(0, pyramid(148))
    for index in QBERT_CUBES:
        ram = pyramid(148)
        ram[index] = 26
        observer.advance(25, ram)
    assert observer.score == 525 and observer.max_initial_cubes == 1
    assert not observer.result(True, False)['episode_success']


def test_breakout_requires_both_walls_and_latches_a_clear_before_cutoff():
    observer = task('Breakout')
    observer.advance(432)
    assert not observer.result(True, False)['episode_success']
    observer.advance(431)
    assert not observer.result(True, False)['episode_success']
    observer.advance(1)
    assert observer.result(False, True)['episode_success']
    partial = observer.result(False, False)
    assert not partial['eligible_completed_episode'] and not partial['episode_success']
    assert partial['first_milestone_frame'] == 3


def test_freeway_requires_25_crossings_and_a_complete_natural_round():
    observer = task('Freeway')
    observer.advance(24)
    assert not observer.result(True, False)['episode_success']
    observer.advance(1)
    assert observer.result(True, False)['episode_success']
    assert not observer.result(False, True)['episode_success']
    assert not observer.result(True, True)['episode_success']
    assert not observer.result(False, False)['episode_success']


@pytest.mark.parametrize('game', ['Pong', 'Boxing'])
def test_match_wins_are_final_outcomes_not_early_leads(game):
    observer = task(game)
    observer.advance(1)
    assert not observer.result(False, False)['episode_success']
    assert not observer.result(True, True)['episode_success']
    assert observer.result(True, False)['episode_success']
    observer.advance(-1)
    assert not observer.result(True, False)['episode_success']
    observer.advance(-1)
    assert not observer.result(True, False)['episode_success']


@pytest.mark.parametrize('reward', [True, None, [], float('nan'), float('inf')])
def test_invalid_rewards_fail(reward):
    with pytest.raises(ValueError, match='finite scalar'):
        task('Breakout').advance(reward)


@pytest.mark.parametrize('ram', [None, [], [0] * 127, [256] * 128, [True] * 128])
def test_qbert_requires_actual_byte_ram(ram):
    with pytest.raises(ValueError, match='128 RAM bytes'):
        task('Qbert').advance(0, ram)


def test_wrong_rom_or_game_fail():
    with pytest.raises(ValueError, match='ROM identity'):
        EpisodeTask('ALE/Qbert-v5', 'wrong')
    with pytest.raises(ValueError, match='ROM identity'):
        EpisodeTask('ALE/Frostbite-v5', ROM_SHA256['ALE/Qbert-v5'])
