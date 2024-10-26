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
INIT_SNAKE_LENGTH : int = 5

# Snake Speed in moves per second
MPS : int = 20 # FPS

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
        self.grid_res       = grid_res
        self.grid_pxl_res   = grid_pxl_res
        self.mps            = mps

        '''  Window Layout 
                            Windows Resolution
                |-----------------------------------------|
                |        Score Board  (Top Offset)        |
                |   -----------------------------------   |
                |   |                                 |   |
                |   |                                 |   |
                |   |                                 |   |
                |   |                                 | R |
                | L |                                 | I |
                | E |           Frame Area            | G |
                | F |                                 | H |
                | T |                                 | T |
                |   |                                 |   |
                |   |                                 |   |
                |   |                                 |   |
                |   -----------------------------------   |
                |             Bottom Offset               |
                |-----------------------------------------|
        '''

        self._offset_frame = pygame_loc ((
            OFFSET_TOP,
            OFFSET_LEFT
        ))

        self._offset_play_area = pygame_loc ((
            OFFSET_TOP  + self.grid_pxl_res,
            OFFSET_LEFT + self.grid_pxl_res
        ))

        self._offset_score = pygame_loc ((
            0,
            OFFSET_LEFT
        ))

        self._res_frame = pygame_loc ((
            self.grid_res[0] * self.grid_pxl_res,
            self.grid_res[1] * self.grid_pxl_res
        ))

        self._res_play = (
            self._res_frame[0] - 2 * self.grid_pxl_res,
            self._res_frame[1] - 2 * self.grid_pxl_res
        )

        self._res_win = (
            self._res_frame[0] + OFFSET_LEFT + OFFSET_RIGHT,
            self._res_frame[1] + OFFSET_TOP  + OFFSET_BOTTOM
        )

        self._res_score_brd = pygame_loc ((
            OFFSET_TOP,
            self._res_win[0]
        ))

        self._unit_size = (
            self.grid_pxl_res,
            self.grid_pxl_res
        )

        # pygame setup
        pygame.init()
        self.window = pygame.display.set_mode(self._res_win)
        pygame.display.set_caption(game_name)
        # self.clock = pygame.time.Clock()

        # Snake Body surface
        self._snake_body_surf = pygame.Surface(self._unit_size)
        self._snake_body_surf.fill(COLOR.SNAKE_BODY.value)

        # Snake Head surface
        self._snake_head_surf = pygame.Surface(self._unit_size)
        self._snake_head_surf.fill(COLOR.SNAKE_HEAD.value)

        # Snake tracker
        self.snake = deque()
        # Direction tracker
        self.movement_direction : str = 'UP'

        # Food surface
        self._food_surf = pygame.Surface(self._unit_size)
        self._food_surf.fill(COLOR.FOOD.value)
        # food tracker
        self.food = ()

        # background surface
        self._bg_surf = pygame.Surface(self._unit_size)
        self._bg_surf.fill(COLOR.BG.value)

        # Wall / Frame Border
        self._wall_border_surf = pygame.Surface(self._res_frame)
        self._wall_border_surf.fill(COLOR.WALL.value)
        self._play_surf = pygame.Surface(self._res_play)
        self._play_surf.fill(COLOR.BG.value)

        self._wall_border_surf.blit(self._play_surf, self._unit_size)

        
        # Score Board
        self._score_board_surf = pygame.Surface(self._res_score_brd)
        self._score_board_surf.fill(COLOR.BG.value)
        self._score_font       = pygame.font.SysFont('arial', OFFSET_TOP, bold=True)
        # self.font = pygame.font.Font('arial.ttf', top_offset)


        self.reset()

    def reset (self) -> None:

        #! reset scoreboard
        self.score = 0
        self._update_scoreboard()

        #! Reset frame area
        #* 1. Initiate border walls
        #* 2. Initialise Snake and initial movement direction
        #* 3. Initialise Food

        # create wall
        self._create_wall()
        
        # create snake and initial movement direction
        self._create_snake()

        # create food
        self._create_food()

        # refresh window
        self._refresh()

    def step_snake(self) -> bool:

        # Calculate new position
        new_position : np.ndarray = self.snake[0] + DIRECTIONS[self.movement_direction]

        # if head touches snake or wall in next postion then end the game
        if self._is_snake (new_position) or self._is_wall (new_position):
            # print("\n\nGame Over\n\n------------Quiting------------\n\n")
            return False
        
        # if new position is not food, remove tail
        if self._is_food (new_position.copy()):
            self._create_food()
            self.score += 1
            self._update_scoreboard()
        else:
            self._draw_sqr(self.snake.pop(), COLOR.BG.name)

        # change previous head to tail
        self._draw_sqr(self.snake[0], COLOR.SNAKE_BODY.name)
        # add new position as head
        self._draw_sqr(new_position.copy(), COLOR.SNAKE_HEAD.name)

        self._refresh()
        return True

    def run (self) -> None:
        time : int = perf_counter_ns()
        NS_PER_FRAME : float = 1e+9 / MPS # nano second per frame 

        while self._handle_keystroke() and not self._quit_event():
            if ((perf_counter_ns() - time) - NS_PER_FRAME) > -1e3:
                time = perf_counter_ns()    # Reset timer
                # self.clock.tick()
                if not self.step_snake():
                    break

                # print('time per frame: ', int((perf_counter_ns() - time) * 1e-3), ' micro s')
                # print('Frame Rate: ', self.clock.get_fps())
        pygame.quit()


    def _update_scoreboard (self) -> None:
        # Redraw scoreboard
        score_text = self._score_font.render(
            f'Score: {self.score}', 
            True,  
            COLOR.SCORE.value
        )

        # erase score board
        self._score_board_surf.fill(COLOR.BG.value)

        # draw new score
        self._score_board_surf.blit(score_text, self._offset_score)

        self.window.blit(self._score_board_surf, (0, 0))
        # pygame.display.flip() # initial frame

    def _create_wall (self) -> None:
        # Grid array to track snake food and wall
        self.grid = np.zeros(self.grid_res, dtype=int)

        # initialize wall
        self.grid[:,  0] = GRID_SYMBOL.WALL.value # left boundry
        self.grid[:, -1] = GRID_SYMBOL.WALL.value # right boundry
        self.grid[ 0, :] = GRID_SYMBOL.WALL.value # top boundry
        self.grid[-1, :] = GRID_SYMBOL.WALL.value # bottom boundry

        # reset the play area
        self._play_surf.fill(COLOR.BG.value)

        self.window.blit(self._wall_border_surf, self._offset_frame)
        # pygame.display.flip() # initial frame

    def _create_snake (self) -> None:

        # Take random position for snake head
        #! it is easier if there is no abstruction for snake with only one direction selected
        #! Hence a smaller subset of grid is chosen for snake head placement
        # Grid lenght - snake length - 2 wall 
        snake_head = np.array([
            random.randint(1+INIT_SNAKE_LENGTH, self.grid_res[0]-INIT_SNAKE_LENGTH-2), 
            random.randint(1+INIT_SNAKE_LENGTH, self.grid_res[1]-INIT_SNAKE_LENGTH-2) 
        ], dtype=int)
        self.snake = deque()
        # self.snake.append(snake_head.copy()) #

        # add snake head to play surface
        self._draw_sqr(snake_head.copy(), COLOR.SNAKE_HEAD.name)

        posible_dir_keys = list(DIRECTIONS.keys())      # get list of directions  
        choice_dir = random.choice(posible_dir_keys)    # choose a key at random
        
        snake_body = snake_head.copy()
        for _ in range(INIT_SNAKE_LENGTH-1):
            snake_body += DIRECTIONS[choice_dir]
            # self.snake.append(snake_body.copy())
            self._draw_sqr(snake_body.copy(), COLOR.SNAKE_BODY.name)
        
        posible_dir_keys.remove(choice_dir)
        self.movement_direction = random.choice(posible_dir_keys)

        # self.window.blit(self._play_surf, self._offset_play_area)
        # pygame.display.flip() # initial frame

    def _create_food (self) -> None:
        while True: 
            self.food = np.array([
                random.randint(1, self.grid_res[0]-2),
                random.randint(1, self.grid_res[1]-2)
            ])
            if not self._is_snake(self.food):
                break

        # draw food to play area
        self._draw_sqr(self.food, COLOR.FOOD.name)

        # self.window.blit(self._play_surf, self._offset_play_area)
        # pygame.display.flip() # initial frame

    def _get_pix (self, coord : tuple[int, int]) -> tuple[int, int]:
        return pygame_loc((
            (coord[0] - 1) * self.grid_pxl_res, 
            (coord[1] - 1) * self.grid_pxl_res
        ))

    def _is_snake (self, coord : tuple[int, int]) -> bool:
        if type(coord) == np.ndarray:
            coord = tuple(coord)
        return True if self.grid[coord] == GRID_SYMBOL.SNAKE.value else False
    
    def _is_wall (self, coord : tuple[int, int]) -> bool:
        if type(coord) == np.ndarray:
            coord = tuple(coord)
        return True if self.grid[coord] == GRID_SYMBOL.WALL.value else False
    
    def _is_food (self, coord : tuple[int, int]) -> bool:
        if type(coord) == np.ndarray:
            coord = tuple(coord)
        return True if self.grid[coord] == GRID_SYMBOL.FOOD.value else False

    def _draw_sqr (self, coord : tuple[int, int], typ : str):

        if typ == 'SNAKE_HEAD' :
            # Append coordinates to the snake dequeue
            # ! appendleft because its a head
            self.snake.appendleft(coord)

            # Put snake head to play area
            self._play_surf.blit(
                self._snake_head_surf, 
                self._get_pix(coord)
            )

            # change grid symbol
            self.grid[tuple(coord)] = GRID_SYMBOL['SNAKE'].value

        if typ == 'SNAKE_BODY' :
            # Append coordinates to the snake dequeue
            if len(self.snake) == 1:
                self.snake.append(coord)

            elif not (np.array_equal(self.snake[0], coord) or np.array_equal(self.snake[1], coord)):
                self.snake.append(coord)

            # Put snake body to play area
            self._play_surf.blit(
                self._snake_body_surf, 
                self._get_pix(coord)
            )

            # change grid symbol
            self.grid[tuple(coord)] = GRID_SYMBOL['SNAKE'].value

        if typ == 'BG' :
            # # remove snake body from snake dequeue
            # if np.array_equal(self.snake[-1], coord):
            #     self.snake.pop()

            # change back to background
            self._play_surf.blit(
                self._bg_surf,
                self._get_pix(coord)
            )

            # change grid symbol
            self.grid[tuple(coord)] = GRID_SYMBOL[typ].value

        if typ == 'FOOD' :
            # change back to background
            self._play_surf.blit(
                self._food_surf,
                self._get_pix(coord)
            )

            # change grid symbol
            self.grid[tuple(coord)] = GRID_SYMBOL[typ].value

    def _refresh(self) -> None:
        self.window.blit(self._play_surf, self._offset_play_area)
        pygame.display.flip() # initial frame

    def _quit_event(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True 
        
        return False
    
    def _handle_keystroke(self) -> bool:
        # * Updates direction depending on the user input
         
        key_pressed = pygame.key.get_pressed() # user input dictionary

        if key_pressed [K_x]:
            # print("\n\n------------Quiting------------\n\n from 'X' key\n\n")
            return False
        
        if key_pressed [K_UP] or key_pressed [K_w]:
            # print("\n====== Pressed K_UP ======\n")
            # move UP if the direction in not DOWN
            if self.movement_direction == 'DOWN':
                return True
            self.movement_direction = 'UP'
            # print(self.direction, "\n")

        if key_pressed [K_DOWN] or key_pressed [K_s]:
            # print("\n====== Pressed K_DOWN ======\n")
            # move DOWN if the direction in not UP
            if self.movement_direction == 'UP':
                return True
            self.movement_direction = 'DOWN'
            # print(self.direction, "\n")

        if key_pressed [K_LEFT] or key_pressed [K_a]:
            # print("\n====== Pressed K_LEFT ======\n")
            # move LEFT if the direction in not RIGHT
            if self.movement_direction == 'RIGHT':
                return True
            self.movement_direction = 'LEFT'
            # print(self.direction, "\n")

        if key_pressed [K_RIGHT] or key_pressed [K_d]:
            # print("\n====== Pressed K_RIGHT ======\n")
            # move RIGHT if the direction in not LEFT
            if self.movement_direction == 'LEFT':
                return True
            self.movement_direction = 'RIGHT'
            # print(self.direction, "\n")

        return True
    
   



if __name__ == '__main__' :
    SnakeGame(GRID_RESOLUTION, GRID_PIXEL_RESOLUTION, MPS).run()
