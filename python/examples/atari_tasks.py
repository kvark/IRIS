"""CPU evaluation observers for the pinned five-game ROMs; never policy inputs."""

import math


ROM_SHA256 = {
    'ALE/Pong-v5': '41623e3c26148f96cbe77d56512845d224a63c8fe5bf90e38527155e39ec96d3',
    'ALE/Boxing-v5': '462ab7dae012a175763c4ce88ac7a20d23e8fb68b7125e97c474e0696ed40d95',
    'ALE/Freeway-v5': '3ef620234b98f22bfd7c06d2467bac9d208e763c9f4309452384d59a741ba893',
    'ALE/Breakout-v5': '376323f051c3c373c887fd83abead39d87d844ff283d435f4addbfc1710c6fd5',
    'ALE/Qbert-v5': '3257221832a7607696b06b9985103c12a4c24c61b0c655e8f52aed8e38194b76',
}
TASKS = {
    'ALE/Pong-v5': 'natural_match_win',
    'ALE/Boxing-v5': 'natural_match_win',
    'ALE/Freeway-v5': '25_crossings_in_a_natural_round',
    'ALE/Breakout-v5': 'both_walls_864_points',
    'ALE/Qbert-v5': 'first_pyramid_all_21_cubes',
}
# Display-color addresses from OCAtari 99c87467, verified against this ROM.
QBERT_CUBES = (21, 52, 54, 83, 85, 87, 98, 100, 102, 104,
               1, 3, 5, 7, 9, 32, 34, 36, 38, 40, 42)


class EpisodeTask:
    def __init__(self, environment, rom_sha256):
        if environment not in ROM_SHA256 or rom_sha256 != ROM_SHA256[environment]:
            raise ValueError('unsupported task or ROM identity')
        self.environment = environment
        self.frames = 0
        self.score = 0.0
        self.first_milestone_frame = None
        self.initial_pyramid_seen = False
        self.max_initial_cubes = 0

    def advance(self, reward, ram=None):
        if type(reward) not in (int, float) or not math.isfinite(reward):
            raise ValueError('reward must be a finite scalar')
        self.frames += 1
        self.score += reward
        if not math.isfinite(self.score):
            raise ValueError('non-finite episode score')
        reached = False
        if self.environment == 'ALE/Qbert-v5':
            if ram is None or len(ram) != 128 or any(type(value) is not int or not 0 <= value <= 255 for value in ram):
                raise ValueError('Qbert observer requires 128 RAM bytes')
            colors = [ram[index] for index in QBERT_CUBES]
            self.initial_pyramid_seen |= all(color == 148 for color in colors)
            if self.initial_pyramid_seen and self.first_milestone_frame is None:
                self.max_initial_cubes = max(self.max_initial_cubes, colors.count(26))
                reached = self.max_initial_cubes == 21
        elif self.environment == 'ALE/Breakout-v5':
            reached = self.score >= 864
        elif self.environment == 'ALE/Freeway-v5':
            reached = self.score >= 25
        if reached and self.first_milestone_frame is None:
            self.first_milestone_frame = self.frames

    def result(self, terminated, truncated):
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError('boundary flags must be boolean')
        completed = terminated or truncated
        natural = terminated and not truncated
        if self.environment in ('ALE/Pong-v5', 'ALE/Boxing-v5'):
            success = natural and self.score > 0
        elif self.environment == 'ALE/Freeway-v5':
            success = natural and self.score >= 25
        else:
            success = completed and self.first_milestone_frame is not None
        return dict(task=TASKS[self.environment], episode_success=success,
                    eligible_completed_episode=completed, terminated=terminated, truncated=truncated,
                    episode_score=self.score, episode_frames=self.frames,
                    first_milestone_frame=self.first_milestone_frame,
                    max_initial_qbert_cubes=self.max_initial_cubes if self.environment == 'ALE/Qbert-v5' else None)
