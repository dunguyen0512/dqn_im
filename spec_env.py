import gymnasium as gym
from gymnasium import error, spaces, utils
# from gym.utils import seeding
import numpy as np
import os

from load_data import load_spec_im
import matplotlib.pyplot as plt

MAX_STEPS = 5 ## should be 20
# WINDOW_SIZE = 7
WINDOW_SIZE = 56 #agent window size
RANDOM_LOC = False
CENTER_START = False  # start at center of the image
BOTTOM_START = True  # start at bottom-center of the image
ALPHA_PENALTY = 0.1  # penalty for incorrect prediction


'''
Notes:
    Agent navigation for image classification. We propose
    an image classification task starting with a masked image
    where the agent starts at a random location on the image. It
    can unmask windows of the image by moving in one of 4 directions: 
    {UP, DOWN, LEFT, RIGHT}. At each timestep it
    also outputs a probability distribution over possible classes
    C. The episode ends when the agent correctly classifies the
    image or a maximum of 10 steps is reached. The agent receives a 
    -0.1 reward at each timestep that it misclassifies the
    image. The state received at each time step is the full image
    with unobserved parts masked out.

    -- for now, agent outputs direction of movement and class prediction (0-3)
    -- correct guess ends game
'''

class SpecEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, type='train', seed=2069,train_class_counts=None):
        
        if seed:
            np.random.seed(seed=seed)
        (x_train, y_train), (x_test, y_test),class_weights = load_spec_im("spectrogram_datasetBL", image_size=(224,224))
        # (x_train, y_train), (x_test, y_test),class_weights = load_spec_im("spectrogram_dataset", image_size=(224,224))
        if type == 'train':
            if train_class_counts is not None:
                self.X, self.Y = self._subsample_by_class(x_train, y_train, train_class_counts)
            else:
                self.X = x_train
                self.Y = y_train
                # self.n = len(y_train)
            
        elif type == 'test':
            self.X = x_test
            self.Y = y_test
            self.n = len(y_test)
        else:
            raise ValueError("Invalid type: %s. Use 'train' or 'test'." % type)
        
        self.n = len(self.Y) # number of images
        h, w = self.X[0].shape
        # print("Image shape: %d x %d" % (h, w))
        self.h = h // WINDOW_SIZE
        self.w = w // WINDOW_SIZE
        
        self.mask = np.zeros((h, w))
        self.alpha_penalty = float(ALPHA_PENALTY)
        self.class_weights = self._compute_class_weights(self.Y)
        # direction of the mask move: up/ down
        # prediction 0 or 1 for H and F
        # see 'step' for interpretation
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(0, 255, [h, w])
        
    def step(self, action):
        # action a consists of
        #   1. direction in {N, S, E, W}, determined by = a (mod 4)
        #   2. predicted class (0-4), determined by floor(a / 4)
        assert(self.action_space.contains(action))
        dir, Y_pred = action % 4, action // 4
        self.steps += 1
        move_map = {
            0: [-1, 0], # T
            1: [1, 0],  # B
            2: [0, 1],  # L
            3: [0, -1]  # R
        }
                
        new_pos = self.pos + move_map[dir]
        valid_move = (0 <= new_pos[0] < self.h) and (0 <= new_pos[1] < self.w)
        if valid_move:
            self.pos = new_pos  # stop at edge by ignoring invalid moves
        self._reveal()
        # state (observation) consists of masked image (h x w)
        obs = self._get_obs()

        ## calculate reward function based on weighted criteria
        y_true = self.Y[self.i]
        w_true = self.class_weights.get(y_true, 1.0)
        # print('Trueweight:',w_true)
        if Y_pred == y_true:
            reward = w_true 
        else:
            reward = -1.0 * self.alpha_penalty * w_true

    # (optional) small penalty for trying to leave
        if valid_move:
            reward += 0.04
        else:
            reward -= 0.1
        # game ends if prediction is correct or max steps is reached
        done = Y_pred != self.Y[self.i] or self.steps >= MAX_STEPS
        
        # info is empty (for now)
        info = { 'true_class': int(y_true),
        'pred_class': int(Y_pred),
        'class_weight': float(w_true),}  
        return obs, reward, done, info

    def _subsample_by_class(self, X, Y, counts_dict):
        """Return subset with up to counts_dict[class_id] samples from each class."""
        self.num_classes = 4
        chosen_idx = []
        for c in range(self.num_classes):
            idx_c = np.where(Y == c)[0]
            np.random.shuffle(idx_c)
            k = counts_dict.get(c, None)
            if k is None:
                chosen_idx.extend(idx_c)  # keep all
            else:
                chosen_idx.extend(idx_c[:min(k, len(idx_c))])
        chosen_idx = np.array(chosen_idx)
        np.random.shuffle(chosen_idx)
        return X[chosen_idx], Y[chosen_idx]


    def _compute_class_weights(self, Y):
        """Inverse-frequency weights normalized to mean=1 for present classes."""
        num_classes = 4
        counts = {c: int(np.sum(Y == c)) for c in range(num_classes)}
        inv = {c: (1.0 / n if n > 0 else 0.0) for c, n in counts.items()}
        present = [v for v in inv.values() if v > 0]
        mean_inv = (sum(present) / len(present)) if present else 1.0
        return {c: (v / mean_inv if mean_inv > 0 else 1.0) for c, v in inv.items()}    
    def reset(self):
        # resets the environment and returns initial observation
        # zero the mask, move to random location, and choose new image
        
        self.i = np.random.randint(self.n)
        # initialize at random location or image center
        if RANDOM_LOC:
            self.pos = np.array([np.random.randint(self.h), 
                                 np.random.randint(self.w)])
        elif CENTER_START:
            r_center = ((self.mask.shape[0] - 1) // 2) // WINDOW_SIZE
            c_center = ((self.mask.shape[1] - 1) // 2) // WINDOW_SIZE
            self.pos = np.array([int(r_center), int(c_center)], dtype=int)
        elif BOTTOM_START:
            # self.pos = np.array([self.h - 1,  np.random.randint(self.w-1)], dtype=int)
            ##random bottom center left /right 
            c_left = (self.w - 1) // 2
            c_right = self.w // 2
            c = np.random.choice([c_left, c_right]) if c_left != c_right else c_left
            self.pos = np.array([self.h - 1, int(c)], dtype=int)
        else:
            self.pos = np.array([int(self.h // 2), int(self.w // 2)], dtype=int)

        self.mask[:, :] = 0
        self._reveal()

        self.steps = 0
        
        return self._get_obs()
        
    def _get_obs(self):
        obs = (self.X[self.i] * self.mask ).astype(np.float32)
        assert self.observation_space.contains(obs)
        return obs.astype(np.uint8)
        
    def _reveal(self):
        # reveal the window at self.pos
        # h, w = self.pos
        # r_grid, c_grid = int(self.pos[0]), int(self.pos[1])
        # r0 = r_grid * WINDOW_SIZE
        # r1 = (r_grid + 1) * WINDOW_SIZE
        # c0 = c_grid * WINDOW_SIZE
        # c1 = (c_grid + 1) * WINDOW_SIZE

        # H_full, W_full = self.mask.shape
        # r0 = max(0, min(r0, H_full))
        # r1 = max(0, min(r1, H_full))
        # c0 = max(0, min(c0, W_full))
        # c1 = max(0, min(c1, W_full))

        # if r0 < r1 and c0 < c1:
        #     self.mask[r0:r1, c0:c1] = 1
        h, w = self.pos
        h_low, h_high = h * WINDOW_SIZE, (h + 1) * WINDOW_SIZE
        w_low, w_high = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
        
        self.mask[h_low:h_high, w_low:w_high] = 1

        
    def render(self, mode='human', close=False, save_path=None, show=True, dpi=150):
        fig, axs = plt.subplots(1, 3, figsize=(3.6, 2), constrained_layout=True)
        fig.suptitle(f"Step {self.steps}", y=0.99)  # push title up a bit

        axs[0].imshow((self.mask * 255).astype(np.uint8), interpolation="nearest")
        axs[0].set_title("Mask", pad=2); axs[0].axis("off")

        axs[1].imshow(self.X[self.i], interpolation="nearest")
        axs[1].set_title("Image", pad=2); axs[1].axis("off")

        axs[2].imshow(self.X[self.i] * self.mask, interpolation="nearest")
        axs[2].set_title("Unmasked", pad=2); axs[2].axis("off")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
        (plt.show() if show else plt.close(fig))


