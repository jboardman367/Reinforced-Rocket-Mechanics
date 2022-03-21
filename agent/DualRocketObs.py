from typing import Any, List

import numpy as np
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.common_values import BOOST_LOCATIONS

large_boost_mask = np.array(BOOST_LOCATIONS)[:, 2] < 72


def _add_player(obs: List, player: PlayerData, inverted: bool):
    # Add physics data
    car = player.inverted_car_data if inverted else player.car_data
    obs.extend(car.forward())
    obs.extend(car.right())
    obs.extend(car.up())
    obs.extend(car.linear_velocity / 2300)
    obs.extend(car.position / 5000)
    obs.extend(car.angular_velocity / 5)

    # Other data
    obs.extend([
        player.has_flip,
        player.on_ground,
        player.boost_amount
    ])


def _add_blank_player(obs: List):
    # This will have as much impossible data as I can fit
    obs.extend([0, 0, 0])
    obs.extend([0, 0, 0])
    obs.extend([0, 0, 0])
    obs.extend([0, 0, -1])
    obs.extend([0, 0, -1])
    obs.extend([0, 0, 0])

    # Other data
    obs.extend([
        -1,
        -1,
        -1
    ])


class DualRocketObs(ObsBuilder):
    def __init__(self):
        super().__init__()
        self.boost_timers = None

    def reset(self, initial_state: GameState):
        self.boost_timers = np.zeros(len(initial_state.boost_pads))

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        inverted = player.team_num == 1
        obs = []
        # TODO maybe stack other player frames later? Not every frame, but enough to get an idea.
        # probably more relevant once some kind of pool is introduced.

        # Add self
        _add_player(obs, player, inverted)
        # Add allies
        for p in state.players:
            if p.team_num == player.team_num and not p is player:
                _add_player(obs, p, inverted)
        # Add blank allies
        for _ in range(2 - len(state.players) // 2):
            _add_blank_player(obs)
        # Add opponents
        for p in state.players:
            if p.team_num != player.team_num and not p is player:
                _add_player(obs, p, inverted)
        # Add blank opponents
        for _ in range(3 - len(state.players) // 2):
            _add_blank_player(obs)

        # Add the ball
        ball = state.inverted_ball if inverted else state.ball
        obs.extend(ball.position / 5000)
        obs.extend(ball.linear_velocity / 2300)
        obs.extend(ball.angular_velocity / 5)

        # BOOST (copied from Necto)
        if player.car_id == 0:
            boost =  state.inverted_boost_pads if inverted else state.boost_pads
            new_grabs = (boost == 1) & (self.boost_timers == 0)
            self.boost_timers[new_grabs] = 0.4 + 0.6 * large_boost_mask
            self.boost_timers *= boost
            obs.extend(self.boost_timers)
            self.boost_timers -= 8 / 1200
            self.boost_timers[self.boost_timers < 0] = 0
        else:
            obs.extend(self.boost_timers)

        # TODO copy demo code from Necto too

        return obs

