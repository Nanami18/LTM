import gymnasium as gym
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
        id="MiniGrid-ObjLocateS13New",
        entry_point="envs.findingobj_minigrid_new:MiniGrid_ObjLocateS13New",
        kwargs={"size":13, "agent_view_size":3}
    )


# An environment where the agent has to find a ball given the instruction that specify the color
class MiniGrid_ObjLocateS13New(MiniGridEnv):

    def __init__(self, size=13, **kwargs):
        
        self.colors = list(COLORS.keys())
        self.obj_loc = {}
        self.total_reward = 0
        mission_space = MissionSpace(mission_func=self._gen_mission, ordered_placeholders=[COLOR_NAMES])
        
        super().__init__(
            mission_space=mission_space,
            height=size,
            width=size,
            max_steps=500,
            see_through_walls=False,
            **kwargs
        )
    
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
        self.target_pos = self.obj_loc[target_color]

        self.mission = f'locate the {target_color} ball'

        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)


        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        print(action)


        if action == self.actions.toggle:
            if (ax == tx and ay == ty):
                # print("Reached the target pos")
                reward = 1
                while self.target_pos == (tx, ty):
                    # print("Reached the target pos")
                    target_color = self.np_random.choice(self.colors)
                    self.target_pos = self.obj_loc[target_color]
                    self.mission = f'locate the {target_color} ball'
                    
                    
        self.total_reward += reward

        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.total_reward = 0

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}
    
    def compute_expert_action(self):
        tx, ty = self.target_pos
        goal_positions = set([(tx, ty)])
        
        # compute shortest path
        agent_pose = self.agent_pos[0], self.agent_pos[1], self.agent_dir
        open_set = [(agent_pose, [])]
        closed_set = set()


        print("goal_positions = ", goal_positions)
        print("open set = ", open_set)
        
        while True:
            try:
                pose, path = open_set.pop(0)
            except IndexError:
                #image = Image.fromarray(self.render('rgb_array'))
                #image.save('./no_path.png')
                #raise ExpertException(
                #    'No path to goal. This should never happen.')
                # this actually can happen if the agent moves a ball in the way
                # of the goal
                path = [self.actions.done]
                break
                
            closed_set.add(pose)
            x, y, direction = pose
            if (x,y) in goal_positions:
                path.append(self.actions.toggle)
                break
            # left
            # breakpoint()
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
        
# return first action in the shortest path