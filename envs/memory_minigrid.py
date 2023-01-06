import gymnasium as gym
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
        id="MiniGrid-MemoryS13-v0-seeobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
    )
    gym.register(
        id="MiniGrid-MemoryS9-v0-seeobj",
        entry_point="envs.memory_minigrid:MiniGrid_MemoryS13_v0_seeobj",
        kwargs={"size":9}
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
        **kwargs,
    ):
        self.random_length = random_length
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
        self.grid.set(1, height // 2 - 1, start_room_obj('green'))

        other_objs = self._rand_elem([[Ball, Key], [Key, Ball]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, other_objs[0]('green'))
        self.grid.set(*pos1, other_objs[1]('green'))

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)
        
        # self.place_agent(top=(2,4), size=(1,1))

        self.mission = 'go to the matching object at the end of the hallway'