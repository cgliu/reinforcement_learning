import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import matplotlib
import pygame

# Observation: [x, y, x_desired, y_desired, dx, dy]
# Action: [force_x, force_y]
num_points = 101
num_lookahead = 10
class PathTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode):
        super(PathTrackingEnv, self).__init__()
        self.dt = 0.1
        self.max_steps = 100
        self.mass = 1.0
        self.step_count = 0
        self.window_size = 1024  # The size of the PyGame window

        # State: x, y, vx, vy
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3 + num_lookahead,), dtype=np.float32)
        # Action: force in x and y direction
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.ref_path =  np.zeros((4, num_points))
        self.ref_path[0, :] = np.linspace(0, 100, num_points)
        self.ref_path[3, :] = 1.0
        assert self.ref_path.shape[0] == 4 and self.ref_path.shape[1] == num_points  

        self.render_mode = render_mode
        self.window = None
        self.resolution = 10 # pixel per meter
        self.clock = None

    def reset(self, seed = 0):
        # s: x, y, heading
        # a: heading change rate
        self.state = np.random.rand(3) 
        self.state[2] -= 0.5
        
        # self.state = np.zeros(4)
        self.step_count = 0
        return self._get_obs(), {}

    def _desired_path(self, t):
        x_d = t * 0.5
        y_d = np.sin(0.1 * t * self.dt * 10)
        dx_d = 0.5
        dy_d = 0.1 * np.cos(0.1 * t * self.dt * 10) * 10
        return np.array([x_d, y_d, dx_d, dy_d])

    def _get_obs(self):
        x, y, theta = self.state
        #
        ego = np.array([x, y]).reshape(-1, 1)
        dist = np.linalg.norm(self.ref_path[:2, :] - ego, axis = 0)
        min_index = np.argmin(dist)

        return np.array([x, y, theta] + dist[min_index: min_index + num_lookahead].tolist(), dtype = np.float32)
        
        # Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
        #                [np.sin(theta),  np.cos(theta), 0],
        #                [0.0, 0.0, 1.0]])
        # T =  np.eye(4)
        # T[:3,:3] = Rz
        # T[:2, 3] = np.array([x, y])
        # T_inv = np.linalg.inv(T)
        
        # path_in_robot = T_inv @ self.ref_path
        # path_in_robot = path_in_robot[:2, :]
        # dist = np.linalg.norm(path_in_robot, axis = 0) # (num_points,)
        # # assert dist.shape[0] == num_points 
        
        # min_index = np.argmin(dist)
        # path_lookahead = path_in_robot[:, min_index:min_index + num_lookahead]
        
        # return np.array([x, y, theta] + path_lookahead.flatten().tolist(), dtype = np.float32)
    
    def step(self, action):
        obs = self._get_obs()
        x, y, theta = obs[:3]
        lookahead = obs[3:]
        
        swirl = np.clip(action, -1, 1)
        theta += swirl * self.dt * 10.0
        theta = theta.item()

        vx = np.cos(theta)  * 1.0
        vy = np.sin(theta)  * 1.0

        x += vx * self.dt
        y += vy * self.dt

        self.state = np.array([x, y, theta])
        self.step_count += 1

        reward = (- np.sum(lookahead ** 2) / num_lookahead
                  - 0.0005 * action.item() ** 2)
        
        # done = self.step_count >= self.max_steps
        next_obs = self._get_obs()
        beyond_ref = len(next_obs) < num_lookahead + 3
        terminated =  self.step_count > self.max_steps or beyond_ref
        truncated = self.step_count > self.max_steps or beyond_ref

        return next_obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if mode == 'human':
            self._render_frame()

    def _to_pygame_frame(self, coord):
        T_wfr = np.eye(4)
        T_wfr[1, 1] = -1
        T_wfr[:2, 3] = np.array([self.window_size / 2 / self.resolution, self.window_size /2 / self.resolution])
        return (T_wfr[:2,:2] @ coord  + T_wfr[:2, 3:4]) * self.resolution
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        obs = self._get_obs()
        x = obs[0]
        y = obs[1]
        theta = obs[2]
        lookahead = obs[3:].reshape(2, -1)
        lookahead = self._to_pygame_frame(lookahead)
        points = [tuple(lookahead[:, i]) for i in range(num_lookahead)]
        pygame.draw.lines(
            canvas,
            (0,0,0),
            closed = False,
            points = points,
            width=3,
        )
        
        # First we draw the target
        ego = self._to_pygame_frame(np.array([0,0]).reshape(-1,1))
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                tuple(ego.squeeze().tolist()),
                (self.resolution * 1, self.resolution * 1),
            ),
        )
        
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

env = PathTrackingEnv(render_mode = "human")
check_env(env)  # Optional: validate
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_tracking_tensorboard/")

def test_render():
    env.reset()
    while True:
        env.render(mode = "human")
    
def train():
    model.learn(total_timesteps=30000,  tb_log_name="first_run")    
    model.save("ppo_tracking")
    
def test():
    model.load("ppo_tracking")
    obs, _ = env.reset()
    total_reward = 0
    positions = []
    rewards = []
    heading = []
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        rewards.append(reward)
        positions.append(obs[:2])
        heading.append(np.linalg.norm(obs[2]))
        # env.render(mode = "human")
        if done:
            break

    # return
    positions = np.array(positions)
    
    # Plot tracking result
    _, axs = plt.subplots(3, 1, figsize = (20, 20))
    
    axs[0].plot(positions[:, 0], positions[:, 1], label="Agent Path")
    axs[0].plot(env.ref_path[0,:], env.ref_path[1,:], label="Reference Path", linestyle='--')


    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title("Path Tracking with RL (PPO)")
    axs[0].legend()
    axs[0].axis("equal")
    axs[0].grid(True)

    axs[1].plot(rewards)
    axs[1].set_title("reward vs time")

    axs[2].plot(heading)
    axs[2].set_ylim([0, 2])
    axs[2].set_title("Heading vs time")
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action='store_true', help = "Train")

    args = parser.parse_args()
    if args.train: 
        train()
    else:
        test()
        # test_render()
