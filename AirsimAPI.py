import airsim
import numpy as np
from collections import deque
from gym import spaces
from PIL import Image
class AirsimEnv():

    def __init__(self):
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()
        # input space.
        high = np.array([np.inf, np.inf, 1., 1.])
        low = np.array([-np.inf, -np.inf, 0., 0.])
        self.observation_space = spaces.Box(low=low, high=high)
        # action space: [steer, accel, brake]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        self.default_action = [0.0, 1.0, 0.0]
        # store vehicle speeds
        self.max_speed = 3e5
        self.prev_speed_sample = 40
        self.past_vehicle_speeds = deque([self.max_speed] * self.prev_speed_sample,
                                         maxlen=self.prev_speed_sample)
        self.done=False
        self.lower_speed_limit = 5
        #Convert Airsim image to numpy

    def get_image(self,image):
        image1d = np.fromstring(image.image_data_uint8, dtype=np.uint8)
        image_rgb = image1d.reshape(image.height, image.width, 3)
        return image_rgb

    def process_image(self, image):
        IMAGE_HEIGHT = 128
        IMAGE_WIDTH = 256
        # image=image[int(image.shape[0]/2):]
        im = Image.fromarray(image).resize(size=(IMAGE_WIDTH,IMAGE_HEIGHT))
        image=np.array(im)
        image=np.array((image - np.min(image)) / (np.max(image) - np.min(image)))
        image_trans=np.transpose(np.array(image),(2,0,1))
        return image_trans

    #Take a step in environment
    def step(self,action):
         all_states=[]
         self.car_controls.throttle=np.clip(action[0], 0.0, 1.0)
         self.car_controls.steering = np.clip(action[1], -1.0, 1.0)
         self.car_controls.brake = np.clip(action[2], 0.0, 1.0)
         self.client.setCarControls(self.car_controls)
         car_state = self.client.getCarState()
         response = self.client.simGetImages([airsim.ImageRequest("CenterCamera", airsim.ImageType.Scene, False, False)])[0]
         # scene vision image in uncompressed RGB array
         im = self.get_image(response)
         imP = self.process_image(im)
         self.done = self.client.simGetCollisionInfo().has_collided
         all_states.append(im)
         all_states.append(imP)
         all_states.append(car_state.speed)
         return  all_states,0,self.done

    def reset(self):
        self.past_vehicle_speeds = deque([self.max_speed] * self.prev_speed_sample,
                                         maxlen=self.prev_speed_sample)
        self.done=False
        self.client.reset()
        all_states=[]
        car_state = self.client.getCarState()
        # state_vec = self.sim_state_to_env(car_state)
        response = self.client.simGetImages([airsim.ImageRequest("CenterCamera", airsim.ImageType.Scene, False, False)])[0]
        # scene vision image in uncompressed RGB array
        im = self.get_image(response)
        imP=self.process_image(im)
        all_states.append(im)
        all_states.append(imP)
        all_states.append(self.client.getCarState().speed)
        return all_states




