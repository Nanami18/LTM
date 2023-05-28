import random

import gymnasium as gym
from gymnasium import spaces
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
    DIR_TO_VEC
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
        id="MiniGrid-MemoryS37-v0-seeobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
        kwargs={"size":37}
    )
    gym.register(
        id="MiniGrid-MemoryS31-v0-seeobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
        kwargs={"size":31}
    )
    gym.register(
        id="MiniGrid-MemoryS25-v0-seeobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
        kwargs={"size":25}
    )
    gym.register(
        id="MiniGrid-MemoryS19-v0-seeobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
        kwargs={"size":19}
    )
    gym.register(
        id="MiniGrid-MemoryS13-v0-seeobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
    )
    gym.register(
        id="MiniGrid-MemoryS9-v0-seeobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
        kwargs={"size":9}
    )
    gym.register(
        id="MiniGrid-MemoryS5-v0-seeobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
        kwargs={"size":5}
    )
    gym.register(
        id="MiniGrid-MemoryS5-v0-seeobj-myopic",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
        kwargs={"size":5, "agent_view_size":3, "max_steps_coeff":1}
    )
    gym.register(
        id="MiniGrid-MemoryS7-v0-seeobj-myopic",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
        kwargs={"size":7, "agent_view_size":5, "max_steps_coeff":2}
    )
    gym.register(
        id="MiniGrid-MemoryS9-v0-hallwayobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_hallwayobj",
        kwargs={"size":9}
    )
    gym.register(
        id="MiniGrid-MemoryS9-v0-almosthallwayobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_hallwayobj",
        kwargs={"size":9, "almost_hallway":True}
    )
    gym.register(
        id="MiniGrid-MemoryS13-v0-almosthallwayobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_hallwayobj",
        kwargs={"size":13, "almost_hallway":True}
    )
    gym.register(
        id="MiniGrid-MemoryS19-v0-almosthallwayobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_hallwayobj",
        kwargs={"size":19, "almost_hallway":True}
    )
    gym.register(
        id="MiniGrid-MemoryS9-scalarobs",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_scalarobs",
        kwargs={"size":9}
    )
    gym.register(
        id="MiniGrid-MemoryS13-scalarobs",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_scalarobs",
        kwargs={"size":13}
    )
    gym.register(
        id="MiniGrid-MemoryS19-scalarobs",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_scalarobs",
        kwargs={"size":19}
    )
    gym.register(
        id="MiniGrid-MemoryS25-scalarobs",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_scalarobs",
        kwargs={"size":25}
    )
    gym.register(
        id="MiniGrid-MemoryS31-scalarobs",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_scalarobs",
        kwargs={"size":31}
    )
    gym.register(
        id="MiniGrid-MemoryS37-scalarobs",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_scalarobs",
        kwargs={"size":37}
    )


class MiniGrid_MemoryS13_v0_seeobj(MemoryEnv):
    """
    This environment is a memory test. The agent starts in a small room
    where it sees an object. It then has to go through a narrow hallway
    which ends in a split. At each end of the split there is an object,
    one of which is the same as the object in the starting room. The
    agent has to remember the initial object, and go to the matching
    object at split.
    """

    def __init__(
        self,
        size=13,
        random_length=False,
        max_steps_coeff = 5,
        **kwargs,
    ):
        self.random_length = random_length
        super().__init__(
            size=size,
            max_steps=max_steps_coeff*size**2,
            # Set this to True for maximum speed
            # see_through_walls=False, # The MemoryEnv doesn't take this argument, and always pass false to super(), so we can't set it here
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_end = self._rand_int(4, width - 2)
        else:
            hallway_end = width - 3

        # Start room
        for i in range(1, 5):
            self.grid.set(i, upper_room_wall, Wall())
            self.grid.set(i, lower_room_wall, Wall())
        self.grid.set(4, upper_room_wall + 1, Wall())
        self.grid.set(4, lower_room_wall - 1, Wall())

        # Horizontal hallway
        for i in range(5, hallway_end):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())

        # Vertical hallway
        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_end, j, Wall())
            self.grid.set(hallway_end + 2, j, Wall())

        # Fix the player's start position and orientation
        self.agent_pos = (1, height // 2)
        self.agent_dir = 0

        # Place objects
        start_room_obj = self._rand_elem([Key, Ball])
        if start_room_obj == Key:
            self.grid.set(1, height // 2 - 1, start_room_obj('green'))
        else:
            self.grid.set(1, height // 2 - 1, start_room_obj('blue'))

        other_objs = self._rand_elem([[Ball, Key], [Key, Ball]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        if other_objs[0] == Key:
            self.grid.set(*pos0, other_objs[0]('green'))
        else:
            self.grid.set(*pos0, other_objs[0]('blue'))
        if other_objs[1] == Key:
            self.grid.set(*pos1, other_objs[1]('green'))
        else:
            self.grid.set(*pos1, other_objs[1]('blue'))

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)
        
        # self.place_agent(top=(2,4), size=(1,1))

        self.mission = 'go to the matching object at the end of the hallway'

    def compute_expert_action(self):
        tx, ty = self.success_pos
        goal_positions = set([(tx, ty)])
        
        # compute shortest path
        agent_pose = self.agent_pos[0], self.agent_pos[1], self.agent_dir
        open_set = [(agent_pose, [])]
        closed_set = set()
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
                path.append(self.actions.done)
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
    
    def compute_random_action(self):
        return random.randint(0, 2)

# Make the target object location deterministic
class MiniGrid_MemoryS13_v0_hallwayobj(MemoryEnv):
    """
    This environment is a memory test. The agent starts in a small room
    where it sees an object. It then has to go through a narrow hallway
    which ends in a split. At each end of the split there is an object,
    one of which is the same as the object in the starting room. The
    agent has to remember the initial object, and go to the matching
    object at split.
    """

    def __init__(
        self,
        size=13,
        random_length=False,
        almost_hallway=False,
        **kwargs,
    ):
        self.random_length = random_length
        self.almost_hallway = almost_hallway
        super().__init__(
            size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            # see_through_walls=False, # The MemoryEnv doesn't take this argument, and always pass false to super(), so we can't set it here
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_end = self._rand_int(4, width - 2)
        else:
            hallway_end = width - 3

        # Start room
        for i in range(1, 5):
            self.grid.set(i, upper_room_wall, Wall())
            self.grid.set(i, lower_room_wall, Wall())
        self.grid.set(4, upper_room_wall + 1, Wall())
        self.grid.set(4, lower_room_wall - 1, Wall())

        # Horizontal hallway
        for i in range(5, hallway_end):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())

        # Vertical hallway
        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_end, j, Wall())
            self.grid.set(hallway_end + 2, j, Wall())

        # Fix the player's start position and orientation
        self.agent_pos = (1, height // 2)
        self.agent_dir = 0

        # Place objects
        start_room_obj = self._rand_elem([Key, Ball])
        color_map = {Key : 'green', Ball : 'blue'}
        self.grid.set(1, height // 2 - 1, start_room_obj(color_map[start_room_obj]))
        # self.grid.set(2, height // 2 - 1, start_room_obj(color_map[start_room_obj]))
        # self.grid.set(3, height // 2 - 1, start_room_obj(color_map[start_room_obj]))
        if not self.almost_hallway:
            self.grid.set(4, height // 2 - 1, start_room_obj(color_map[start_room_obj]))
            self.grid.set(5, height // 2 - 1, start_room_obj(color_map[start_room_obj]))
            self.grid.set(6, height // 2 - 1, start_room_obj(color_map[start_room_obj]))

        self.grid.set(1, height // 2 + 1, start_room_obj(color_map[start_room_obj]))
        # self.grid.set(2, height // 2 + 1, start_room_obj(color_map[start_room_obj]))
        # self.grid.set(3, height // 2 + 1, start_room_obj(color_map[start_room_obj]))
        if not self.almost_hallway:
            self.grid.set(4, height // 2 + 1, start_room_obj(color_map[start_room_obj]))
            self.grid.set(5, height // 2 + 1, start_room_obj(color_map[start_room_obj]))
            self.grid.set(6, height // 2 + 1, start_room_obj(color_map[start_room_obj]))

        other_objs = self._rand_elem([[Ball, Key]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, other_objs[0](color_map[other_objs[0]]))
        self.grid.set(*pos1, other_objs[1](color_map[other_objs[1]]))

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)
        
        # self.place_agent(top=(2,4), size=(1,1))

        self.mission = 'go to the matching object at the end of the hallway'

    def compute_expert_action(self):
        tx, ty = self.success_pos
        goal_positions = set([(tx, ty)])
        
        # compute shortest path
        agent_pose = self.agent_pos[0], self.agent_pos[1], self.agent_dir
        open_set = [(agent_pose, [])]
        closed_set = set()
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
                path.append(self.actions.done)
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

# Make the target object location deterministic
class MiniGrid_MemoryS13_scalarobs(MemoryEnv):
    """
    This environment is a memory test. The agent starts in a small room
    where it sees an object. It then has to go through a narrow hallway
    which ends in a split. At each end of the split there is an object,
    one of which is the same as the object in the starting room. The
    agent has to remember the initial object, and go to the matching
    object at split.
    """

    def __init__(
        self,
        size=13,
        random_length=False,
        almost_hallway=False,
        **kwargs,
    ):
        self.random_length = random_length
        self.almost_hallway = almost_hallway
        self.target_color = None
        super().__init__(
            size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            # see_through_walls=False, # The MemoryEnv doesn't take this argument, and always pass false to super(), so we can't set it here
            **kwargs,
        )
        
        self.observation_space = spaces.Discrete(3)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_end = self._rand_int(4, width - 2)
        else:
            hallway_end = width - 3

        # Start room
        for i in range(1, 5):
            self.grid.set(i, upper_room_wall, Wall())
            self.grid.set(i, lower_room_wall, Wall())
        self.grid.set(4, upper_room_wall + 1, Wall())
        self.grid.set(4, lower_room_wall - 1, Wall())

        # Horizontal hallway
        for i in range(5, hallway_end):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())

        # Vertical hallway
        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_end, j, Wall())
            self.grid.set(hallway_end + 2, j, Wall())

        # Fix the player's start position and orientation
        self.agent_pos = (1, height // 2)
        self.agent_dir = 0

        # Place objects
        start_room_obj = self._rand_elem([Key, Ball])
        color_map = {Key : 'green', Ball : 'blue'}
        self.grid.set(1, height // 2 - 1, start_room_obj(color_map[start_room_obj]))
        self.grid.set(1, height // 2 + 1, start_room_obj(color_map[start_room_obj]))

        other_objs = self._rand_elem([[Ball, Key]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, other_objs[0](color_map[other_objs[0]]))
        self.grid.set(*pos1, other_objs[1](color_map[other_objs[1]]))

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
            self.target_color = 1 
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)
            self.target_color = 2
        
        # self.place_agent(top=(2,4), size=(1,1))

        self.mission = 'go to the matching object at the end of the hallway'
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        return self.target_color, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        return 0, reward, terminated, truncated, info

    def compute_expert_action(self):
        tx, ty = self.success_pos
        goal_positions = set([(tx, ty)])
        
        # compute shortest path
        agent_pose = self.agent_pos[0], self.agent_pos[1], self.agent_dir
        open_set = [(agent_pose, [])]
        closed_set = set()
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
                path.append(self.actions.done)
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