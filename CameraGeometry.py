import numpy as np


def CalculateInstrinsic(FOV, image_width, image_height):
    FOV = FOV * np.pi / 180
    focal = (image_width / 2.0) / np.tan(FOV / 2.)
    Cu = image_width / 2.0
    Cv = image_height / 2.0
    return np.array([[focal, 0, Cu],
                     [0, focal, Cv],
                     [0, 0, 1.0]])


def project_polyline(polyline_world, trafo_world_to_cam, K):
    x, y, z = polyline_world[:, 0], polyline_world[:, 1], polyline_world[:, 2]
    homvec = np.stack((x, y, z, np.ones_like(x)))
    proj_mat = K @ trafo_world_to_cam[:3, :]
    pl_uv_cam = (proj_mat @ homvec).T
    u = pl_uv_cam[:, 0] / pl_uv_cam[:, 2]
    v = pl_uv_cam[:, 1] / pl_uv_cam[:, 2]
    return np.stack((u, v)).T


class CameraGeometry(object):

    def __init__(self, height=1.3, pitchDeg=5, imageWidth=1024, imageHeight=512,FOV=45):
        # scalar constants
        self.height = height
        self.pitchDeg = pitchDeg
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.FOV = FOV
        # camera intriniscs and extrinsics
        self.intrinsicMat = CalculateInstrinsic(FOV, imageWidth, imageHeight)
        self.invIntrMatrix = np.linalg.inv(self.intrinsicMat)
        ## Note that "rotation_cam_to_road" has the math symbol R_{rc} in the book
        pitch = pitchDeg * np.pi / 180
        cpitch, spitch = np.cos(pitch), np.sin(pitch)
        ##Camera to road transforms
        self.rotCam_to_Road = np.array([[1, 0, 0], [0, cpitch, spitch], [0, -spitch, cpitch]])
        self.traCam_to_Road = np.array([0, -self.height, 0])
        self.transCam_to_Road = np.eye(4)
        self.transCam_to_Road[0:3, 0:3] = self.rotCam_to_Road
        self.transCam_to_Road[0:3, 3] =self.traCam_to_Road
        ##Road to camera transforms
        self.transRoad_to_Cam = np.eye(4)
        self.transRoad_to_Cam[0:3, 0:3] = self.rotCam_to_Road.T
        self.transRoad_to_Cam[0:3, 3] = -self.traCam_to_Road
        # compute vector nc. Note that R_{rc}^T = R_{cr}
        self.roadNormInCamera =  self.rotCam_to_Road.T @ np.array([0, 1, 0]) ##(Road2Camera rotation matrix X road normal in road coordinates)
        self.roadNormInCamera=self.roadNormInCamera[:,np.newaxis]
        self.traCam_to_Road=self.traCam_to_Road[:,np.newaxis]

    def CamToRoad(self,vec3d):
        return (self.rotCam_to_Road @ vec3d.T + self.traCam_to_Road).T

    def ConvertToiso8855(self,vec3d):
        a=np.roll(vec3d,1,axis=1)
        a[:,1:]*=-1
        return a

    ##Transform 2D pixel points to 3D coordinates w.r.t camera
    def PixelToCamera(self, vec2D):
        ##vec2D   Nx2
        vec2D=np.roll(vec2D,1,axis=1)  ##Change the cols of vector to become (x,y), normally (row(y),col(x))
        o=np.ones((vec2D.shape[0],1))
        new2D=np.concatenate((vec2D,o),axis=1) ##Nx3
        new2D=new2D.T ##3xN
        temp= self.invIntrMatrix @ new2D
        denominator = self.roadNormInCamera.T @ temp
        return (self.height * temp / denominator).T ##Nx3
