import numpy as np
import utils
from utils import UP, DOWN, LEFT, RIGHT

class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        if state is None or action is None:
            return

        (food_dir_x, food_dir_y,
         adjoining_wall_x, adjoining_wall_y,
         adjoining_body_top, adjoining_body_bottom,
         adjoining_body_left, adjoining_body_right) = state

        self.N[food_dir_x][food_dir_y][adjoining_wall_x][adjoining_wall_y][
            adjoining_body_top][adjoining_body_bottom][
            adjoining_body_left][adjoining_body_right][action] += 1


    def update_q(self, s, a, r, s_prime):
        if s is None or a is None:
            return

        n_sa = self.N[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][a]
        alpha = self.C / (self.C + n_sa)

        if s_prime is None:
            max_q_prime = 0  # Terminal state
        else:
            next_q_values = [
                self.Q[s_prime[0]][s_prime[1]][s_prime[2]][s_prime[3]]
                [s_prime[4]][s_prime[5]][s_prime[6]][s_prime[7]][next_a]
                for next_a in self.actions
            ]
            max_q_prime = max(next_q_values)

        current_q = self.Q[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][a]

        self.Q[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][a] = (
                current_q + alpha * (r + self.gamma * max_q_prime - current_q)
        )


    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)
        print(s_prime)
        # TODO - MP12: write your function here

        return utils.RIGHT

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment
        snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y = environment

        food_dir_x = 0 if food_x == snake_head_x else 1 if food_x < snake_head_x else 2
        food_dir_y = 0 if food_y == snake_head_y else 1 if food_y < snake_head_y else 2

        left_wall = snake_head_x == 1 or ((snake_head_x - 2) == rock_x and snake_head_y == rock_y)
        right_wall = snake_head_x == self.display_width - 2 or ((snake_head_x + 2) == rock_x and snake_head_y == rock_y)
        top_wall = snake_head_y == 1 or (snake_head_x == rock_x and (snake_head_y - 1) == rock_y)
        bottom_wall = snake_head_y == self.display_height - 2 or (snake_head_x == rock_x and (snake_head_y + 1) == rock_y)

        if left_wall:
            adjoining_wall_x = 1
        elif right_wall:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0

        # Determine adjoining_wall_y
        if top_wall:
            adjoining_wall_y = 1
        elif bottom_wall:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        # Check for snake body in adjacent squares
        adjoining_body_top = 1 if (snake_head_x, snake_head_y - 1) in snake_body else 0
        adjoining_body_bottom = 1 if (snake_head_x, snake_head_y + 1) in snake_body else 0
        adjoining_body_left = 1 if (snake_head_x - 1, snake_head_y) in snake_body else 0
        adjoining_body_right = 1 if (snake_head_x + 1, snake_head_y) in snake_body else 0

        return (food_dir_x, food_dir_y,
                adjoining_wall_x, adjoining_wall_y,
                adjoining_body_top, adjoining_body_bottom,
                adjoining_body_left, adjoining_body_right)
