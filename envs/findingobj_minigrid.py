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
    gym.register(
        id="MiniGrid-ObjLocate-Empty-S9",
        entry_point="envs.findingobj_minigrid:MiniGrid_ObjLocate_Empty",
        kwargs={"size":9, "agent_view_size":7}
    )
    gym.register(
        id="MiniGrid-ObjLocate-Empty-S13",
        entry_point="envs.findingobj_minigrid:MiniGrid_ObjLocate_Empty",
        kwargs={"size":13, "agent_view_size":7}
    )
    gym.register(
        id="MiniGrid-ObjLocateEasy-S13",
        entry_point="envs.findingobj_minigrid:MiniGrid_ObjLocateEasyS13",
        kwargs={"size":13, "agent_view_size":7}
    )
    gym.register(
        id="MiniGrid-ObjLocateEasy-S9",
        entry_point="envs.findingobj_minigrid:MiniGrid_ObjLocateEasyS13",
        kwargs={"size":9, "agent_view_size":7}
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

    def ColoredWall(self, color):
        class CWall(Wall):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs, color=color)
        return CWall

    def maze_gen(self, width, height):
        self.grid.horz_wall(2, 2, 6, obj_type=self.ColoredWall(color='green'))
        self.grid.horz_wall(4, 4, 4, obj_type=self.ColoredWall(color='yellow'))
        self.grid.vert_wall(9, 2, 5, obj_type=self.ColoredWall(color='red'))
        self.grid.horz_wall(2, 6, 6, obj_type=self.ColoredWall(color='blue'))
        self.grid.horz_wall(2, 9, 3, obj_type=self.ColoredWall(color='purple'))
        self.grid.horz_wall(7, 9, 3, obj_type=self.ColoredWall(color='green'))
        self.grid.vert_wall(6, 7, 3, obj_type=self.ColoredWall(color='yellow'))
        self.grid.vert_wall(4, 10, 2, obj_type=self.ColoredWall(color='purple'))

        self.grid.vert_wall(2, 1, 4, obj_type=self.ColoredWall(color='red'))

    def compute_expert_action(self):
        tx, ty = self.target_pos
        goal_positions = set([(tx, ty)])
        
        # compute shortest path
        agent_pose = self.agent_pos[0], self.agent_pos[1], self.agent_dir
        open_set = [(agent_pose, [])]
        closed_set = set()
        while True:
            try:
                pose, path = open_set.pop(0)
            except IndexError:
                path = [self.actions.toggle]
                break
                
            closed_set.add(pose)
            x, y, direction = pose
            if (x,y) in goal_positions:
                path.append(self.actions.toggle)
                break

            left_pose = (x, y, (direction-1)%4)
            if left_pose not in closed_set:
                left_path = path[:]
                left_path.append(self.actions.left)
                open_set.append((left_pose, left_path))
            
            # right
            right_pose = (x, y, (direction+1)%4)
            if right_pose not in closed_set:
                right_path = path[:]
                right_path.append(self.actions.right)
                open_set.append((right_pose, right_path))
            
            # forward
            vx, vy = DIR_TO_VEC[direction]
            fx = x + vx
            fy = y + vy
            forward_cell = self.grid.get(fx, fy)
            if forward_cell is None or forward_cell.can_overlap():
                forward_pose = (fx, fy, direction)
                if forward_pose not in closed_set:
                    forward_path = path[:]
                    forward_path.append(self.actions.forward)
                    open_set.append((forward_pose, forward_path))
        
        return path[0]


class MiniGrid_ObjLocate_Empty(MiniGrid_ObjLocateS13):
    def maze_gen(self, width, height):
        return
        
class MiniGrid_ObjLocateEasyS13(MiniGrid_ObjLocateS13):
    def maze_gen(self, width, height):
        self.grid.horz_wall(1, height//3, (width-3)//2, obj_type=self.ColoredWall(color='yellow'))
        self.grid.horz_wall((width-3)//2+1, height//3, (width-3)-((width-3)//2), obj_type=self.ColoredWall(color='red'))
        self.grid.horz_wall(1, height-height//3, (width-3)//2, obj_type=self.ColoredWall(color='blue'))
        self.grid.horz_wall((width-3)//2+1, height-height//3, (width-3)-((width-3)//2), obj_type=self.ColoredWall(color='purple'))