import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras, MeshRenderer, MeshRasterizer
from pytorch3d.renderer import  HardPhongShader, TexturesVertex, DirectionalLights
from FaceModel import FaceModel
from Individu import Individu
from Rendu import Rendu

class Model(nn.Module):
    def __init__(self, n_shape, n_color, path_model='model2019_fullHead.h5',
                 path_image='Photos/VisageCheveux.png',
                 path_contour=None,
                 path_predictor="shape_predictor_68_face_landmarks.dat", device='cuda:0'):  
        super().__init__()
        #Composantes de notre modèle
        self.Face = FaceModel(n_shape, n_color, path=path_model, device=device)
        self.Indiv = Individu(path_image, path_contour, path_predictor, device=device)
        n = self.Indiv.image_size
        self.Render = Rendu([n,n], device)
        self.n_shape = n_shape
        self.n_color = n_color
        self.device = device
        self.Land_BFM = [
                            43632, 58036, 31522, 49084, 57409, 46869, 30691, 30559, 37693, 11565,
                            9050, 13602, 20884, 19788, 10861, 17485, 26018, 33645, 36710, 46080,
                            56307, 43916, 4521, 27012, 20125, 15034, 3206, 27734, 16684, 38126,
                            15848, 43817, 45898, 26991, 4900, 7883, 43885, 43056, 51725, 44916,
                            42143, 51219, 5380, 4689, 13761, 27786, 8177, 6778, 32562, 38117,
                            56274, 38232, 8686, 1872, 17193, 26456, 22167, 33237, 51462, 48496,
                            32242, 48601, 33241, 27358, 3335, 20076, 58180, 33800
                         ]

        #Hyperparamètres de la fonction de perte
        self.wlan = 10
        self.wc = 1.
        self.wregc = 1e-5
        self.wregl = 1e-5
      
    def get_points(self, verts):              
        #projetter les points correspondant aux landmarks dans le plan de camera
        points = verts[:, self.Land_BFM, :]
        n = self.Indiv.image_size
        image_size = torch.tensor([[n,n]]).to(self.device)
        cameras = FoVPerspectiveCameras(device=self.device, R=self.Render.compute_R(), T=self.Render.T)
        cordinates = cameras.transform_points_screen(points, image_size).to(self.device)
        return cordinates

    def loss_land(self, verts):
        points = self.get_points(verts) 
        points = points[:, :, 0:2]          # projected the points of the model
        distance = torch.sum((points - self.Indiv.landmarks) ** 2)         #calculate the distance
        return distance / 68

    def deformation(self, param_shape):
        deformation = torch.multiply(self.Face.PCA_basis, param_shape).reshape(self.n_shape,-1,3).cuda()
        deformed_face = self.Face.verts_mean + torch.sum(deformation, axis=0)
        return deformed_face
    
    def loss_color(self, image):
        distance = torch.sum((image[0,...,:3] - self.Indiv.image_contour) ** 2) 
        return distance / self.Indiv.card

    def loss_reg(self):
        S = 0
        S += torch.sum((self.Face.param_shape*self.Face.PCA_reg)**2)*self.wregl
        S += torch.sum((self.Face.param_color*self.Face.Color_reg)**2)*self.wregc
        return S

    def loss(self):
        S = 0
        S += self.wlan*self.loss_land(self.Face.verts)
        S += self.wc*self.loss_color(self.Face.image)
        S += self.loss_reg()
        return S

    def render_visual(self):
        n = self.Indiv.image_size
        image_size = torch.tensor([[n, n]]).to(self.device) 
        image = self.Face.image
        plt.imshow(image[0, ..., :3].cpu().detach().numpy())
        points = self.Face.verts[:, self.Land_BFM, :]

        cordinates = self.Render.cameras.transform_points_screen(points, image_size=image_size).to(self.device)
        plt.scatter(cordinates[0,...,0].cpu().detach().numpy(),
                    cordinates[0,...,1].cpu().detach().numpy(), color='r')
        plt.scatter(self.Indiv.landmarks[0,...,0].cpu().detach().numpy(),
                    self.Indiv.landmarks[0,...,1].cpu().detach().numpy())

        plt.show()

    def render(self):
        image = self.Face.image
        plt.imshow(image[0,...,:3].cpu().detach().numpy())
        plt.show()  

    def forward(self):
        #Calcul la forme et la couleur en fonction des paramètres 
        R = self.Render.compute_R()
        self.Render.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=self.Render.T)
        self.Face.verts = self.deformation(self.Face.param_shape)
        self.Face.colors = self.Face.color_mean + \
                            (self.Face.param_color@self.Face.Color_basis).reshape(1,-1,3).to(self.device)
        self.Face.mesh = Meshes(verts=self.Face.verts, faces=self.Face.triangles,
                                textures=TexturesVertex(verts_features=self.Face.colors)).cuda()

        #Modifie les paramètres de rendu et stocke le rendu
        self.Render.lights = DirectionalLights(device = self.device, direction = self.Render.lights_direction)
        self.Render.renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.Render.cameras,
                                                                      raster_settings=self.Render.raster_settings),
                                            shader=HardPhongShader(device=self.device,
                                                                   cameras=self.Render.cameras,
                                                                   lights=self.Render.lights))
        n = self.Indiv.image_size
        image_size = torch.tensor([n, n]).cuda()
        self.Face.image = self.Render.renderer(self.Face.mesh, 
                                               materials=self.Render.materials, 
                                               image_size=image_size).cuda()

        # Calcule la perte 
        loss = self.loss()
        return loss
    
    def supperposition(self):
        image = np.array(self.Face.image[0,...,:3].detach().cpu())

        res = np.zeros(image.shape)
        ref = np.array(self.Indiv.image_ref.cpu())
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                if sum(image[i][j]) == 3.:
                    res[i][j] = ref[:,:,:3][i][j]

                else: 
                    res[i][j] = image[i][j]
        return res