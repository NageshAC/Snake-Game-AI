import  pygame
import  random 
import  numpy       as      np
from    enum        import  Enum, IntEnum
from    collections import  deque
from    time        import  perf_counter_ns

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_w,
    K_s,
    K_a,
    K_d,
    K_x
)

# Game Window Configuration
GRID_RESOLUTION : tuple[int, int] = (45, 80)

# Pixel Size of each Grid
GRID_PIXEL_RESOLUTION : int = 10 

# Initial Snake Size
SNAKE_LENGTH : int = 5

# Snake Speed in moves per second
MPS : int = 10 # FPS

# Play Area offset
OFFSET_TOP    : int  = 25
OFFSET_BOTTOM : int  = 0
OFFSET_LEFT   : int  = 0
OFFSET_RIGHT  : int  = 0

# Colors used in the game window
class COLOR (Enum):
    WALL        = (200,   0,   0)  # RED   color
    BG          = (  0,   0,   0)  # BLACK color
    FOOD        = (  0, 200,   0)  # GREEN color
    SNAKE_BODY  = (128, 128, 125)  # GRAY  color
    SNAKE_HEAD  = (255, 255, 255)  # WHITE color
    SCORE       = (255, 255, 255)  # WHITE color

# Corresponding Grid Symbols
class GRID_SYMBOL (IntEnum):
    BG     = 0
    WALL   = 1
    FOOD   = 2
    SNAKE  = 3

DIRECTIONS = dict({
    'UP'      : np.array([-1,  0], dtype=int),
    'RIGHT'   : np.array([ 0,  1], dtype=int),
    'DOWN'    : np.array([ 1,  0], dtype=int),
    'LEFT'    : np.array([ 0, -1], dtype=int)
})

# PyGame take the order (Columns X Rows)
def pygame_loc (loc: tuple[int, int] | np.ndarray) -> tuple[int, int]:
    return (loc[1], loc[0])

class SnakeGame :
    def __init__ (self, grid_res:tuple[int,int], grid_pxl_res:int, mps:int, game_name:str = "Snake") -> None :
        pass
    
    def run (self) -> None:
        pass




if __name__ == '__main__' :
    SnakeGame(GRID_RESOLUTION, GRID_PIXEL_RESOLUTION, MPS)