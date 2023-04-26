## code adapted from https://github.com/Stanford-ILIAD/PantheonRL/blob/master/pantheonrl/envs/blockworldgym/gridutils.py

# functions to deal with generating a grid world and gravity
import numpy as np
import warnings

HORIZONTAL = 0
VERTICAL = 1


def pad_grid(grid, new_grid_size):
    new_grid = -1 * np.ones((new_grid_size, new_grid_size))
    new_grid[:grid.shape[0], :grid.shape[1]] = grid
    return new_grid


def generate_random_world(width, num_blocks, num_colors, blocksize):
    gridworld = np.zeros((width, width))
    blocks_placed = 0
    while blocks_placed < num_blocks:
        if drop_random(width, gridworld, num_colors, blocksize) == 1:
            blocks_placed += 1
        else:
            ## Unable to place any more blocks. need to retry generation process
            return None
    return gridworld

def drop_random(width, gridworld, num_colors, blocksize):
    orientation = np.random.randint(2)   ## even for blocksize 1 this is ok
    if orientation == HORIZONTAL: 
        possible_locs = [i for i in range(width - blocksize + 1) if np.sum(gridworld[0, i:i+blocksize]) == 0]
        if possible_locs:
            x = np.random.choice(possible_locs) 
        else:
            ## try VERTICAL orientation
            orientation = VERTICAL
            possible_locs = [i for i in range(width) if np.sum(gridworld[:blocksize, i]) == 0]
            if possible_locs:
                x = np.random.choice(possible_locs) 
            else:
                ## This is possible in weird cases. Unable to place anymore blocks
                return -1
    else:
        possible_locs = [i for i in range(width) if np.sum(gridworld[:blocksize, i]) == 0]
        if possible_locs:
            x = np.random.choice(possible_locs)
        else:
            ## try HORIZONTAL orientation
            orientation = HORIZONTAL
            possible_locs = [i for i in range(width - blocksize + 1) if np.sum(gridworld[0, i:i+blocksize]) == 0]
            if possible_locs:
                x = np.random.choice(possible_locs) 
            else:
                ## This is possible in weird cases. Unable to place anymore blocks
                return -1
    y = gravity(gridworld, orientation, x, blocksize)
    if y == -1:
        raise ValueError("This should not happen!")
        # return -1 # error
    else:
        color = np.random.randint(num_colors) + 1
        place(gridworld, x, y, color, orientation, blocksize)
    return 1

def place(gridworld, x, y, color, orientation, blocksize):
    gridworld[y][x] = color
    if orientation == HORIZONTAL:
        gridworld[y][x + blocksize -1] = color
    if orientation == VERTICAL:
        gridworld[y + blocksize - 1][x] = color
    
def gravity(gridworld, orientation, x, blocksize):
    # check if placeable
    #if gridworld[0][x] != 0:
    #    return -1
    if (orientation == HORIZONTAL and x > len(gridworld)-blocksize):
        return -1
    if (orientation == HORIZONTAL and np.sum(gridworld[0, x:x+blocksize]) != 0) or (orientation == VERTICAL and np.sum(gridworld[:blocksize, x]) != 0):
        return -1
    for y in range(len(gridworld)):
        # this is the final position if it hits something (there's something or a floor right below it)
        if orientation == HORIZONTAL:
            if y == len(gridworld) - 1:
                return y
            if np.sum(gridworld[y + 1, x:x+blocksize]) != 0:
                return y
        if orientation == VERTICAL:
            if y == len(gridworld) - blocksize:
                return y
            if gridworld[y + blocksize, x] != 0:
                return y
    raise ValueError("We shouldn't be able to reach here!")
    # return -1 # shouldn't be able to reach here

def matches(grid1, grid2):
    # number of nonzero elements in the same place
    # we can divide the two, and if two nonzero elements are in the same place the quotient is 1
    # then count nonzeroes in the array quotient==1
    # but i'm filtering the divide by zero warning -- should be careful about this later
    warnings.filterwarnings('ignore', 'invalid value encountered in true_divide')
    warnings.filterwarnings('ignore', 'divide by zero encountered in true_divide')
    return np.count_nonzero((grid1/grid2 == 1))