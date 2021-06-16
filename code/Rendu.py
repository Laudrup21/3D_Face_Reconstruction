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
        """Image_size correspond au format de l'image largeur x hauteur"""
        super().__init__()
        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, )
        self.lights_direction = nn.Parameter(torch.tensor([[-1.0, 0.0, 0.0]]))

        self.lights = DirectionalLights(device=device, direction=self.lights_direction)

        self.diffuse_color = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
        self.ambient_color = nn.Parameter(torch.tensor([[1.0, 1.0, 1.0]]))
        self.shininess = nn.Parameter(torch.tensor([10.0]))

        self.materials = Materials(device=device,specular_color=[[0.0, 0.0, 0.0]])
        self.device = device
        self.image_size = image_size

        self.euler_angles = nn.Parameter(torch.tensor([[3.1354, -0.1492,  3.1634]]).to(device))
        self.T = nn.Parameter(torch.tensor([[-9.5367e-07, -0.0000e+00,  2.8018e+02]], device = device))


        self.cameras=FoVPerspectiveCameras(device=device,R=self.compute_R(),T=self.T)

        self.renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras, 
                                                               raster_settings=self.raster_settings),
                                     shader=HardPhongShader(device=device, 
                                                            cameras=self.cameras, lights=self.lights))
        self.Land_BFM = [
                            43632, 58036, 31522, 49084, 57409, 46869, 30691, 30559, 37693, 11565,
                            9050, 13602, 20884, 19788, 10861, 17485, 26018, 33645, 36710, 46080,
                            56307, 43916, 4521, 27012, 20125, 15034, 3206, 27734, 16684, 38126,
                            15848, 43817, 45898, 26991, 4900, 7883, 43885, 43056, 51725, 44916,
                            42143, 51219, 5380, 4689, 13761, 27786, 8177, 6778, 32562, 38117,
                            56274, 38232, 8686, 1872, 17193, 26456, 22167, 33237, 51462, 48496,
                            32242, 48601, 33241, 27358, 3335, 20076, 58180, 33800
                        ]

    def compute_R(self):
        return euler_angles_to_matrix(self.euler_angles,"XYZ")

    def render(self, mesh):
        """ Crée un rendu de l'objet mesh à partir des paramètres de rendu. """
        image = self.renderer(meshes_world=mesh, R=self.R,T=self.T)
        # Calculate the silhouette loss
        return image

    def compute_projected_lmks(self, mesh):
        """ Compute the (x, y) positions of the 68 face landmarks on the screen space """
        lmks_verts = mesh.verts_padded()[:, self.Land_BFM, :]
        camera = self.cameras
        size=[]
        for i in range(len(lmks_verts)):
            size.append(self.image_size)
            proj_lmks = camera.transform_points_screen(lmks_verts,torch.tensor(size).to(self.device))

        return proj_lmks[0, :, :2]

    def fixed_cam(self, mesh):
        """ Compute the (x, y) positions of the 68 face landmarks on the screen space """
        lmks_verts = mesh.verts_padded()[:, self.Land_BFM, :]
        camera = FoVPerspectiveCameras(device=self.device, R=self.R.detach(), T=self.T.detach())
        size = []
        for i in range(len(lmks_verts)):
            size.append(self.image_size)
            proj_lmks = camera.transform_points_screen(lmks_verts, torch.tensor(size).to(self.device))

        return proj_lmks[0, :, :2]  