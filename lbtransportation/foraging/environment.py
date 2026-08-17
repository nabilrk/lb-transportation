from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
import logging
from typing import Iterable

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class TransportationEnv(gym.Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        min_player_level,
        max_player_level,
        min_food_level,
        max_food_level,
        field_size,
        max_num_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=True,
        grid_observation=False,
        observe_agent_levels=True,
        penalty=0.0,
        render_mode=None,
        curriculum=False,
    ):
        self.logger = logging.getLogger(__name__)
        self.render_mode = render_mode
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty
        self.curriculum = curriculum

        if isinstance(min_food_level, Iterable):
            assert (
                len(min_food_level) == max_num_food
            ), "min_food_level must be a scalar or a list of length max_num_food"
            self.min_food_level = np.array(min_food_level)
        else:
            self.min_food_level = np.array([min_food_level] * max_num_food)

        if max_food_level is None:
            self.max_food_level = None
        elif isinstance(max_food_level, Iterable):
            assert (
                len(max_food_level) == max_num_food
            ), "max_food_level must be a scalar or a list of length max_num_food"
            self.max_food_level = np.array(max_food_level)
        else:
            self.max_food_level = np.array([max_food_level] * max_num_food)

        if self.max_food_level is not None:
            # check if min_food_level is less than max_food_level
            for min_food_level, max_food_level in zip(
                self.min_food_level, self.max_food_level
            ):
                assert (
                    min_food_level <= max_food_level
                ), "min_food_level must be less than or equal to max_food_level for each food"

        self.max_num_food = max_num_food
        self._food_spawned = 0.0

        if isinstance(min_player_level, Iterable):
            assert (
                len(min_player_level) == players
            ), "min_player_level must be a scalar or a list of length players"
            self.min_player_level = np.array(min_player_level)
        else:
            self.min_player_level = np.array([min_player_level] * players)

        if isinstance(max_player_level, Iterable):
            assert (
                len(max_player_level) == players
            ), "max_player_level must be a scalar or a list of length players"
            self.max_player_level = np.array(max_player_level)
        else:
            self.max_player_level = np.array([max_player_level] * players)

        if self.max_player_level is not None:
            # check if min_player_level is less than max_player_level for each player
            for i, (min_player_level, max_player_level) in enumerate(
                zip(self.min_player_level, self.max_player_level)
            ):
                assert (
                    min_player_level <= max_player_level
                ), f"min_player_level must be less than or equal to max_player_level for each player but was {min_player_level} > {max_player_level} for player {i}"

        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation
        self._observe_agent_levels = observe_agent_levels

        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(6)] * len(self.players))
        )
        self.observation_space = gym.spaces.Tuple(
            tuple([self._get_observation_space()] * len(self.players))
        )

        self.viewer = None

        self.n_agents = len(self.players)
        
        self.goal_pos = None
        self.box_lifted_status = {}

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        player_levels = sorted(self.max_player_level)
        max_food_level = (
            max(self.max_food_level)
            if self.max_food_level is not None
            else sum(player_levels[:3])
        )
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_num_food = self.max_num_food

            if self._observe_agent_levels:
                min_obs = [-1, -1, 0, 0] * max_num_food + [-1, -1, 0] * len(self.players) + [-1, -1]
                max_obs = [field_x - 1, field_y - 1, max_food_level, 1] * max_num_food + [
                    field_x - 1,
                    field_y - 1,
                    max(self.max_player_level),
                ] * len(self.players) + [field_x - 1, field_y - 1]
            else:
                min_obs = [-1, -1, 0, 0] * max_num_food + [-1, -1] * len(self.players) + [-1, -1]
                max_obs = [field_x - 1, field_y - 1, max_food_level, 1] * max_num_food + [
                    field_x - 1,
                    field_y - 1,
                ] * len(self.players) + [field_x - 1, field_y - 1]
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            if self._observe_agent_levels:
                agents_max = np.ones(grid_shape, dtype=np.float32) * max(
                    self.max_player_level
                )
            else:
                agents_max = np.ones(grid_shape, dtype=np.float32)

            # foods layer: foods level
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: is the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # lifted layer: is a box lifted
            lifted_min = np.zeros(grid_shape, dtype=np.float32)
            lifted_max = np.ones(grid_shape, dtype=np.float32)

            # goal layer: is it the goal
            goal_min = np.zeros(grid_shape, dtype=np.float32)
            goal_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min, lifted_min, goal_min])
            max_obs = np.stack([agents_max, foods_max, access_max, lifted_max, goal_max])

        low_obs = np.array(min_obs)
        high_obs = np.array(max_obs)
        assert low_obs.shape == high_obs.shape
        return gym.spaces.Box(
            low=low_obs, high=high_obs, shape=[len(low_obs)], dtype=np.float32
        )

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(
            players,
            min_player_level=1,
            max_player_level=2,
            min_food_level=1,
            max_food_level=None,
            field_size=None,
            max_num_food=None,
            sight=None,
            max_episode_steps=50,
            force_coop=False,
        )

        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self, max_num_food, min_levels, max_levels):
        food_count = 0
        attempts = 0
        min_levels = max_levels if self.force_coop else min_levels

        # permute food levels
        food_permutation = self.np_random.permutation(max_num_food)
        min_levels = min_levels[food_permutation]
        max_levels = max_levels[food_permutation]

        while food_count < max_num_food and attempts < 1000:
            attempts += 1
            row = self.np_random.integers(1, self.rows - 1)
            col = self.np_random.integers(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = (
                min_levels[food_count]
                if min_levels[food_count] == max_levels[food_count]
                else self.np_random.integers(
                    min_levels[food_count], max_levels[food_count] + 1
                )
            )
            food_count += 1
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False
        if hasattr(self, 'goal_pos') and self.goal_pos == (row, col):
            return False

        return True

    def spawn_players(self, min_player_levels, max_player_levels):
        # permute player levels
        player_permutation = self.np_random.permutation(len(self.players))
        min_player_levels = min_player_levels[player_permutation]
        max_player_levels = max_player_levels[player_permutation]
        for player, min_player_level, max_player_level in zip(
            self.players, min_player_levels, max_player_levels
        ):
            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.integers(0, self.rows)
                col = self.np_random.integers(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.np_random.integers(min_player_level, max_player_level + 1),
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return player.position[0] > 0
        elif action == Action.SOUTH:
            return player.position[0] < self.rows - 1
        elif action == Action.WEST:
            return player.position[1] > 0
        elif action == Action.EAST:
            return player.position[1] < self.cols - 1
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_num_food):
                obs[4 * i] = -1
                obs[4 * i + 1] = -1
                obs[4 * i + 2] = 0
                obs[4 * i + 3] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[4 * i] = y
                obs[4 * i + 1] = x
                obs[4 * i + 2] = observation.field[y, x]
                obs[4 * i + 3] = 1 if self.box_lifted_status.get((y, x), False) else 0

            player_obs_len = 3 if self._observe_agent_levels else 2
            offset = self.max_num_food * 4
            for i in range(len(self.players)):
                obs[offset + player_obs_len * i] = -1
                obs[offset + player_obs_len * i + 1] = -1
                if self._observe_agent_levels:
                    obs[offset + player_obs_len * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[offset + player_obs_len * i] = p.position[0]
                obs[offset + player_obs_len * i + 1] = p.position[1]
                if self._observe_agent_levels:
                    obs[offset + player_obs_len * i + 2] = p.level
            
            # Goal
            goal_offset = offset + player_obs_len * len(self.players)
            obs[goal_offset] = self.goal_pos[0] if self.goal_pos else -1
            obs[goal_offset + 1] = self.goal_pos[1] if self.goal_pos else -1

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                if self._observe_agent_levels:
                    agents_layer[player_x + self.sight, player_y + self.sight] = (
                        player.level
                    )
                else:
                    agents_layer[player_x + self.sight, player_y + self.sight] = 1

            foods_layer = np.zeros(grid_shape, dtype=np.float32)
            foods_layer[self.sight : -self.sight, self.sight : -self.sight] = (
                self.field.copy()
            )

            access_layer = np.ones(grid_shape, dtype=np.float32)
            # out of bounds not accessible
            access_layer[: self.sight, :] = 0.0
            access_layer[-self.sight :, :] = 0.0
            access_layer[:, : self.sight] = 0.0
            access_layer[:, -self.sight :] = 0.0
            # agent locations are not accessible
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
            # food locations are not accessible
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x + self.sight, y + self.sight] = 0.0
                
            lifted_layer = np.zeros(grid_shape, dtype=np.float32)
            for (r, c), lifted in self.box_lifted_status.items():
                if lifted:
                    lifted_layer[r + self.sight, c + self.sight] = 1.0
            
            goal_layer = np.zeros(grid_shape, dtype=np.float32)
            if self.goal_pos is not None:
                goal_layer[self.goal_pos[0] + self.sight, self.goal_pos[1] + self.sight] = 1.0

            return np.stack([agents_layer, foods_layer, access_layer, lifted_layer, goal_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return (
                agent_x,
                agent_x + 2 * self.sight + 1,
                agent_y,
                agent_y + 2 * self.sight + 1,
            )

        observations = [self._make_obs(player) for player in self.players]
        if self._grid_observation:
            layers = make_global_grid_arrays()
            agents_bounds = [
                get_agent_grid_bounds(*player.position) for player in self.players
            ]
            nobs = tuple(
                [
                    layers[:, start_x:end_x, start_y:end_y]
                    for start_x, end_x, start_y, end_y in agents_bounds
                ]
            )
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations])

        # check the space of obs
        for i, obs in enumerate(nobs):
            assert self.observation_space[i].contains(
                obs
            ), f"obs space error: obs: {obs}, obs_space: {self.observation_space[i]}"

        return nobs

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            # setting seed
            super().reset(seed=seed, options=options)

        self.field = np.zeros(self.field_size, np.int32)
        
        if getattr(self, 'curriculum', False):
            self.goal_pos = (3, 3)
            direction = self.np_random.integers(0, 4)
            if direction == 0: self.field[3,2] = 2
            elif direction == 1: self.field[2,3] = 2
            elif direction == 2: self.field[3,4] = 2
            else: self.field[4,3] = 2
            self._food_spawned = self.field.sum()
        else:
            # Goal position is determined later if needed, but for now we set it randomly
            self.goal_pos = self._get_random_empty_pos()

        self.spawn_players(self.min_player_level, self.max_player_level)
        player_levels = sorted([player.level for player in self.players])
        
        if not getattr(self, 'curriculum', False):
            self.spawn_food(
                self.max_num_food,
                min_levels=self.min_food_level,
                max_levels=self.max_food_level
                if self.max_food_level is not None
                else np.array([sum(player_levels[:3])] * self.max_num_food),
            )

        self.current_step = 0
        self._game_over = False
        
        self.box_lifted_status = {}
        self.delivered_boxes = []
            
        self._gen_valid_moves()

        nobs = self._make_gym_obs()
        return nobs, self._get_info()

    def _get_random_empty_pos(self):
        # Fix the goal diagonally adjacent to the lower right corner
        return (self.rows - 2, self.cols - 2)

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()
        moving_players = {}

        # determine intended movements
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                moving_players[player] = player.position
            elif action == Action.NORTH:
                moving_players[player] = (player.position[0] - 1, player.position[1])
            elif action == Action.SOUTH:
                moving_players[player] = (player.position[0] + 1, player.position[1])
            elif action == Action.WEST:
                moving_players[player] = (player.position[0], player.position[1] - 1)
            elif action == Action.EAST:
                moving_players[player] = (player.position[0], player.position[1] + 1)
            elif action == Action.LOAD:
                moving_players[player] = player.position
                loading_players.add(player)

        # resolve boxes being transported
        for box_pos, is_lifted in list(self.box_lifted_status.items()):
            if not is_lifted:
                continue
            
            frow, fcol = box_pos
            box_level = self.field[frow, fcol]
            if box_level == 0:
                self.box_lifted_status.pop(box_pos, None)
                continue
            
            adj_players = self.adjacent_players(frow, fcol)
            # Find players that were attached to the box
            
            movement_actions = [actions[self.players.index(p)] for p in adj_players]
            
            if len(set(movement_actions)) == 1 and movement_actions[0] in [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]:
                action = movement_actions[0]
                adj_player_level = sum([a.level for a in adj_players])
                if adj_player_level >= box_level:
                    # Move box
                    if action == Action.NORTH: new_box_pos = (frow - 1, fcol)
                    elif action == Action.SOUTH: new_box_pos = (frow + 1, fcol)
                    elif action == Action.WEST: new_box_pos = (frow, fcol - 1)
                    elif action == Action.EAST: new_box_pos = (frow, fcol + 1)
                    
                    # check box collision with walls
                    if 0 <= new_box_pos[0] < self.rows and 0 <= new_box_pos[1] < self.cols:
                        self.field[new_box_pos[0], new_box_pos[1]] = box_level
                        self.field[frow, fcol] = 0
                        self.box_lifted_status.pop(box_pos)
                        self.box_lifted_status[new_box_pos] = True
                    else:
                        # Box hits a wall, drop it
                        self.box_lifted_status[box_pos] = False
                        # Cancel player movements
                        for p in adj_players:
                            moving_players[p] = p.position
                            p.reward -= self.penalty
                else:
                    self.box_lifted_status[box_pos] = False
            else:
                self.box_lifted_status[box_pos] = False
                
        # Move players after boxes
        blocked_players = set()
        
        for player in self.players:
            target = moving_players[player]
            if self.field[target[0], target[1]] > 0:
                blocked_players.add(player)
                
        while True:
            new_blocked = set()
            target_counts = defaultdict(list)
            
            for p in self.players:
                if p in blocked_players:
                    target_counts[p.position].append(p)
                else:
                    target_counts[moving_players[p]].append(p)
                    
            for target, players_targeting in target_counts.items():
                if len(players_targeting) > 1:
                    for p in players_targeting:
                        if p not in blocked_players:
                            new_blocked.add(p)
                            
            if not new_blocked:
                break
            blocked_players.update(new_blocked)
            
        for p in self.players:
            if p in blocked_players:
                p.reward -= self.penalty
            else:
                p.position = moving_players[p]

        # Process new loadings
        while loading_players:
            player = loading_players.pop()
            loc = self.adjacent_food_location(*player.position)
            if loc is None:
                player.reward -= self.penalty
                continue
            
            frow, fcol = loc
            box_level = self.field[frow, fcol]
            if box_level == 0:
                continue

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [p for p in adj_players if p in loading_players or p is player]
            adj_player_level = sum([a.level for a in adj_players])
            loading_players = loading_players - set(adj_players)

            if adj_player_level >= box_level:
                self.box_lifted_status[(frow, fcol)] = True
            else:
                for a in adj_players:
                    a.reward -= self.penalty

        # Check Goal
        if self.goal_pos is not None:
            grow, gcol = self.goal_pos
            if self.field[grow, gcol] > 0:
                self.delivered_boxes.append((grow, gcol, self.field[grow, gcol]))
                self.field[grow, gcol] = 0
                self.box_lifted_status.pop(self.goal_pos, None)
                for p in self.players:
                    p.reward += 1.0
                self.goal_pos = self._get_random_empty_pos()

        # Update valid moves
        self._gen_valid_moves()

        for p in self.players:
            p.reward -= 0.01
            p.score += p.reward

        rewards = [p.reward for p in self.players]
        self._game_over = (self.field.sum() == 0 or self._max_episode_steps <= self.current_step)
        done = self._game_over
        truncated = False
        info = self._get_info()

        return self._make_gym_obs(), rewards, done, truncated, info

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=self.render_mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()

    def test_make_gym_obs(self):
        """Test wrapper to test the current observation in a public manner."""
        return self._make_gym_obs()

    def test_gen_valid_moves(self):
        """Wrapper around a private method to test if the generated moves are valid."""
        try:
            self._gen_valid_moves()
        except Exception as _:
            return False
        return True
