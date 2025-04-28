import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
import cv2

# Define a simple ACC environment
class AdaptiveCruiseControlEnv(gym.Env):
    def __init__(self):
        super(AdaptiveCruiseControlEnv, self).__init__()
        
        # State: [ego_car_speed, distance_to_lead_car, lead_car_speed]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([40.0, 100.0, 40.0]),
            dtype=np.float32
        )
        
        # Action: Acceleration [-3 m/s^2, 3 m/s^2]
        self.action_space = spaces.Box(low=np.array([-3.0]), high=np.array([3.0]), dtype=np.float32)
        
        self.reset()

    def reset(self, seed = 0):
        self.ego_speed = np.random.uniform(20, 30)  # m/s
        self.lead_speed = np.random.uniform(20, 30)  # m/s
        self.distance = np.random.uniform(20, 50)    # meters
        return np.array([self.ego_speed, self.distance, self.lead_speed], dtype=np.float32), {}

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        dt = 0.1  # time step (100 ms)

        # Update ego vehicle
        self.ego_speed = np.clip(self.ego_speed + accel * dt, 0, 40)
        self.distance += (self.lead_speed - self.ego_speed) * dt

        # Lead car random slowdowns
        if np.random.rand() < 0.05:
            self.lead_speed = np.clip(self.lead_speed + np.random.uniform(-0.3, 0), 0, 40)

        # Compute reward
        safe_distance = max(5.0, self.ego_speed * 0.5)  # 0.5 second gap
        distance_error = self.distance - safe_distance

        
        reward = - (distance_error ** 2)  # penalize distance error quadratically
        reward -= 0.1 * (accel ** 2)       # penalize high accelerations (comfort)
        reward = 0.01 * self.ego_speed # reward high speed   
        # Episode termination conditions
        done = False
        if self.distance <= 0:
            reward -= 1000  # crash
            done = True
        if self.distance > 100:
            reward -= 100  # too far
            done = True

        obs = np.array([self.ego_speed, self.distance, self.lead_speed], dtype=np.float32)
        # obs, reward, terminated, truncated, info.
        return obs, reward, done, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Ego Speed: {self.ego_speed:.2f} m/s, Distance: {self.distance:.2f} m")
        elif mode == 'rgb_array':
            cell_size = 5.0
            grid_size = 200
            img_size = int(grid_size * cell_size)
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # White background
            
            # Draw agent
            x, y = grid_size / 2, grid_size / 2
            top_left = (int(x * cell_size), int(y * cell_size))
            bottom_right = (int((x + 4) * cell_size), int((y + 2) * cell_size))
            cv2.rectangle(img,
                          top_left,
                          bottom_right, (0, 255, 0), -1)  # Red agent
            x, y = grid_size / 2 - self.distance, grid_size / 2
            top_left = (int(x * cell_size), int(y * cell_size))
            bottom_right = (int((x + 4) * cell_size), int((y + 2) * cell_size))
            cv2.rectangle(img,
                          top_left,
                          bottom_right, (0, 0, 255), -1)  # Red agent
            self.display_txt(img)
            
            return img
    def display_txt(self, image):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        # fontScale
        fontScale = 0.5
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 1
        # Using cv2.putText() method
        image = cv2.putText(image, f"distance: {self.distance:.2f}", org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
        org = (50, 80)
        image = cv2.putText(image, f"lead_car_speed: {self.lead_speed:.2f}", org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
 
    
# Create environment and check it
env = AdaptiveCruiseControlEnv()
check_env(env)

# Train a SAC agent
model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="./sac_acc_tensorboard/")

# # Train for a while
# model.learn(total_timesteps=100_000)

# # Save the model
# model.save("sac_acc_agent")
# exit(1)


# Train a SAC agent
model.load("sac_acc_agent")

# Enjoy trained agent
obs, _ = env.reset()
cv2.imshow("sdc",env.render('rgb_array'))
# img = plt.imshow(env.render('rgb_array'))
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, _, info = env.step(action)
    image = env.render('rgb_array')
    # img.set_data(env.render('rgb_array'))
    if dones:
        obs, _ = env.reset()
    cv2.imshow("sdc",env.render('rgb_array'))
    cv2.waitKey(1)


