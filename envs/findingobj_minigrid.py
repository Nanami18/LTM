import gymnasium as gym
from gymnasium import spaces

import numpy as np
import random
from minigrid.minigrid_env import (
    MiniGridEnv,
    MissionSpace,
)
from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    DIR_TO_VEC,
    COLOR_NAMES,
)
# Action: 0 left 1 right 2 forward 6 done
from minigrid.core.world_object import (
    Wall,
    Ball,
    Door,
    Key,
)
from minigrid.core.grid import Grid
from minigrid.envs import MemoryEnv

def register_envs():
    gym.register(
        id="MiniGrid-ObjLocateS13",
        entry_point="envs.findingobj_minigrid:MiniGrid_ObjLocateS13",
        kwargs={"size":13, "agent_view_size":3}
    )
    gym.register(
        id="MiniGrid-ObjLocateS13-view5",
        entry_point="envs.findingobj_minigrid:MiniGrid_ObjLocateS13",
        kwargs={"size":13, "agent_view_size":5}
    )
    gym.register(
        id="MiniGrid-ObjLocateS13-view7",
        entry_point="envs.findingobj_minigrid:MiniGrid_ObjLocateS13",
        kwargs={"size":13, "agent_view_size":7}
    )

# An environment where the agent has to find a ball given the instruction that specify the color
class MiniGrid_ObjLocateS13(MiniGridEnv):

    def __init__(self, size=13, **kwargs):
        
        self.colors = list(COLORS.keys())
        self.obj_loc = {}
        mission_space = MissionSpace(mission_func=self._gen_mission, ordered_placeholders=[COLOR_NAMES])
        
        super().__init__(
            mission_space=mission_space,
            height=size,
            width=size,
            max_steps=500,
            see_through_walls=False,
            **kwargs
        )

        self.observation_space['target_color'] = spaces.Discrete(len(self.colors))
    
    @staticmethod
    def _gen_mission(color):
        return 'go to the {} ball'.format(color)
    
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate walls to make a maze
        self.grid.wall_rect(0, 0, width, height)
        self.maze_gen(width, height)

        # Place balls of different colors in random locations
        for color in self.colors:
            ball = Ball(color)
            ball.can_overlap = lambda : True
            ball.can_pickup = lambda : False
            self.place_obj(ball)
            self.obj_loc[color] = ball.cur_pos

        # Set the agent's starting point
        self.agent_pos = self.place_agent()

        # Set the target ball as specified in the mission text
        target_color = self.np_random.choice(self.colors)
        self.target_color = target_color
        self.target_pos = self.obj_loc[target_color]
        self.mission = f'locate the {target_color} ball'

        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        if action == self.actions.toggle:
            if (ax == tx and ay == ty):
                reward = 1
                while self.target_pos == (tx, ty):
                    target_color = self.np_random.choice(self.colors)
                    self.target_color = target_color
                    self.target_pos = self.obj_loc[target_color]
                    self.mission = f'locate the {target_color} ball'

        obs['target_color'] = COLOR_TO_IDX[self.target_color]
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None,):
        obs, info = super().reset(seed=seed, options=options)
        obs['target_color'] = COLOR_TO_IDX[self.target_color]
        return obs, info

    def maze_gen(self, width, height):
        self.grid.horz_wall(2, 2, 6)
        self.grid.horz_wall(4, 4, 4)
        self.grid.vert_wall(9, 2, 5)
        self.grid.horz_wall(2, 6, 6)
        self.grid.horz_wall(2, 9, 3)
        self.grid.horz_wall(7, 9, 3)
        self.grid.vert_wall(6, 7, 3)
        self.grid.vert_wall(4, 10, 2)

        self.grid.vert_wall(2, 1, 4)
        
