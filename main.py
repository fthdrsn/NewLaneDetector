import cv2
import random
from AirsimAPI import *
import torch
from  LaneFollower import model
from  collections import deque
import time
from skimage.measure import label, regionprops
from CameraGeometry import *
from matplotlib import pyplot as plt
from pure_pursuit import *
from scipy.interpolate import interp1d

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
##Orj image
ORJ_IMAGE_HEIGHT = 512
ORJ_IMAGE_WIDTH = 1024
##After preprocessing
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256
IMAGE_CHANNEL = 3
LABEL_CHANNEL = 1
CLASS_NUM = 2
PRETRAINED_PATH="./LaneFollower/pretrained/unetlstm.pth"
FRAME_NUMBER=3 ## Number of consecutive frames

HEIGHT_RATIO=ORJ_IMAGE_HEIGHT/IMAGE_HEIGHT
WIDTH_RATIO=ORJ_IMAGE_WIDTH/IMAGE_WIDTH

###Camera related
FOV=60
CAMERA_HEIGHT=2
PICTH_DEGREE=5
FPS=30

###Load the trained weights for Line detection model (LSTM_Unet)
def LoadModel(model):
    pretrained_dict = torch.load(PRETRAINED_PATH)
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)
    return model

###Get only left and right lines, not other parts
def GetLines(img):
    # Find connected pixels (regions)
    label_im = label(img)
    regions = regionprops(label_im)
    newImg = np.zeros_like(img)
    leftCoord=[]
    rightCoord=[]
    ## Return the first-two regions that have maximum number of pixels (They are usually left and right lines)
    ## To decide whether line is left or not, compare first point's coordinate
    if len(regions) >= 2:
        filledAreas = [r.filled_area for r in regions]
        sortedList = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(filledAreas))]
        sortedList.reverse()
        Left, Right = (regions[sortedList[0]], regions[sortedList[1]]) if regions[sortedList[0]].coords[0, 1] < \
                                                                          regions[sortedList[1]].coords[0, 1] else (
        regions[sortedList[1]], regions[sortedList[0]])
        leftCoord = Left.coords
        rightCoord = Right.coords
        newImg[leftCoord[:, 0], leftCoord[:, 1]] = 1
        newImg[rightCoord[:, 0], rightCoord[:, 1]] = 1
    return newImg,leftCoord,rightCoord

def GetTrajectory(polyLeft, polyRight):
    # trajectory to follow is the mean of left and right lane boundary
    # note that we multiply with -0.5 instead of 0.5 in the formula for y below
    # according to our lane detector x is forward and y is left, but
    # according to Carla x is forward and y is right.
    x = np.arange(-2,10,1.0)
    y = -0.5*(polyLeft(x)+polyRight(x))
    # x,y is now in coordinates centered at camera, but camera is 0.5 in front of vehicle center
    # hence correct x coordinates
    x += 0.5
    traj = np.stack((x,y)).T
    return traj

if __name__=="__main__":

    cameraGeo = CameraGeometry(height=CAMERA_HEIGHT, pitchDeg=PICTH_DEGREE, imageWidth=ORJ_IMAGE_WIDTH,imageHeight=ORJ_IMAGE_HEIGHT, FOV=FOV)
    ##Create a model and load with pretrained weights.
    controller=PurePursuitPlusPID()
    modelUNET =LoadModel(model.UNet_ConvLSTM(IMAGE_CHANNEL,CLASS_NUM).to(DEVICE))
    modelUNET.eval()
    ##Create environment object
    env=AirsimEnv()
    ##To store #FRAME_NUMBER consecutive frames (Lane detector use LSTM, so we need to give consecutive frames  )
    imgBuffer=deque(maxlen=FRAME_NUMBER)
    env.reset()
    throttle = 0
    steer = 0
    brake = 0

    while(True):

        with torch.no_grad():
            action = [throttle, steer, brake]
            next_state, reward, done = env.step(action)
            speed = next_state[2]
            imgBuffer.append(next_state[1])
            if len(imgBuffer)<FRAME_NUMBER:
                continue

            tStr = time.time()
            npArr = np.asarray(imgBuffer)
            bfNumpyT=torch.from_numpy(npArr).unsqueeze(0).to(DEVICE)
            output, feature = modelUNET(bfNumpyT.to(torch.float32))
            pred = output.max(1, keepdim=True)[1]

            ##Expand RGB Image
            # img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
            img = torch.squeeze(pred).cpu().numpy() * 255
            img=img.astype(np.uint8)
            ## Find binary image which has only white lanes and black background and left and right lane coodinates
            newImg,leftCoord,rightCoord=GetLines(img)

            if not isinstance(leftCoord,list):
                if leftCoord.shape[0]>20 and rightCoord.shape[0]>20:
                    ## Convert coordinates to original image size
                    leftCoord[:, 0]*=int(HEIGHT_RATIO)
                    leftCoord[:, 1] *= int(WIDTH_RATIO)
                    rightCoord[:, 0] *= int(HEIGHT_RATIO)
                    rightCoord[:, 1] *= int(WIDTH_RATIO)
                    ##Find 3D pixel coordinates w.r.t camera
                    coordL = cameraGeo.PixelToCamera(leftCoord)
                    coordR = cameraGeo.PixelToCamera(rightCoord)
                    ##Transform points from camera frame to road frame
                    coordInRoadL = cameraGeo.CamToRoad(coordL)
                    coordInRoadR = cameraGeo.CamToRoad(coordR)
                    ##Change koordinates to be convenient with iso8855 coordinate system
                    coordInRoadL = cameraGeo.ConvertToiso8855(coordInRoadL)
                    coordInRoadR = cameraGeo.ConvertToiso8855(coordInRoadR)
                    ##Fitt polynom to left and right road points
                    coeffsLeft = np.polyfit(coordInRoadL[:,0], coordInRoadL[:,1], deg=3)
                    coeffsRight = np.polyfit(coordInRoadR[:, 0], coordInRoadR[:, 1], deg=3)
                    polyLeft=np.poly1d(coeffsLeft)
                    polyRight = np.poly1d(coeffsRight)
                    ##Get the trajectory (center of two polynoms)
                    trajectory=GetTrajectory(polyLeft,polyRight)
                    throttle, steer=controller.get_control(trajectory, speed, desired_speed=3, dt=1./ FPS)
                else:
                    print("NOT ENOUGH POINT")


                # x = np.linspace(0, 60)
                # yl = poly_left(x)
                # yr = poly_right(x)
                # plt.plot(x, yl, label="yl")
                # plt.plot(x, yr, label="yr")
                # plt.xlabel("x (m)")
                # plt.ylabel("y (m)")
                # plt.legend()
                # plt.axis("equal")

                # plt.plot(coordInRoadL[:, 0], coordInRoadL[:, 1], '*')
                # plt.plot(coordInRoadR[:, 0], coordInRoadR[:, 1], '+')
                # plt.show()


            # ## Original black-white
            # imOrj=np.zeros_like(next_state[0])
            # imOrj[leftCoord[:,0],leftCoord[:,1]]=255
            # imOrj[rightCoord[:,0],rightCoord[:,1]]=255
            # cut_v,grid=cameraGeo.precompute_grid()
            # leftRealCoord=FitLines(leftCoord,cut_v,grid)
            # rightRealCoord=FitLines(rightCoord,cut_v,grid)

            # orj=next_state[0]
            # orj[leftCoord[:,0],leftCoord[:,1],:]=0
            # orj[rightCoord[:,0],rightCoord[:,1],:] = 0

            print("Time:", time.time() - tStr)

            if done:
                env.reset()

            cv2.imshow("LEFT",newImg.astype(np.uint8)*255)
            cv2.waitKey(1)




