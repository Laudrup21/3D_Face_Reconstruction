import torch
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix
from tqdm.notebook import tqdm
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.renderer import FoVPerspectiveCameras, FoVOrthographicCameras, PointLights
from pytorch3d.renderer import DirectionalLights, Materials, RasterizationSettings, MeshRenderer
from pytorch3d.renderer import MeshRasterizer, HardPhongShader, TexturesVertex, Textures

class Rendu(nn.Module):
    def __init__(self, image_size, device='cuda:0'):
        """ image_size correspond au format de l'image largeur x hauteur"""
        super().__init__()
        #Paramètres de caméras
        self.euler_angles = nn.Parameter(torch.tensor([[-0.2237,  3.1450, -0.0516]]).to(device))
        self.T = nn.Parameter(torch.tensor([[-6.3663,  -15.7904, 1234.1265]],device = device))
        self.lights_direction = nn.Parameter(torch.tensor([[-1.0, 0.0, 0.0]]))

        #Outils de rendu
        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, )
        self.lights = DirectionalLights(device=device, direction=self.lights_direction)
        self.materials = Materials(device=device, specular_color=[[0.0, 0.0, 0.0]], shininess=10.0)
        self.device = device
        self.image_size = image_size

        self.fov = nn.Parameter(torch.tensor([30.]).to(self.device))
        self.cameras = FoVPerspectiveCameras(fov=self.fov, device=self.device, R=self.compute_R(), T=self.T)

        self.renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras, 
                                                               raster_settings=self.raster_settings),
                                     shader = HardPhongShader(device=device, 
                                                              cameras=self.cameras,lights=self.lights))

    def compute_R(self):
        return euler_angles_to_matrix(self.euler_angles, "XYZ")

    def render(self,mesh):
        image = self.renderer(meshes_world=mesh, R=self.R, T=self.T)
        return image
 