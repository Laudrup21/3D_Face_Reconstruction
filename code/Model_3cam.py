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
from FaceModel_3cam import FaceModel
from Individu import Individu
from Rendu_3cam import Rendu

class Model(nn.Module):
    def __init__(self, n_shape, n_color, path='model2019_fullHead.h5', pathLeft1="Photos/Left.png",
                 pathLeft2=None, pathMiddle1="Photos/Middle.png", pathMiddle2=None, 
                 pathRight1="Photos/Right.png", pathRight2=None, 
                 path7="shape_predictor_68_face_landmarks.dat", device='cuda:0'):  
        super().__init__()
        #Composantes de notre modèle
        self.Face = FaceModel(n_shape, n_color, path=path,device=device)

        self.Indiv1 = Individu(pathLeft1,pathLeft2,path7,device=device)
        self.Indiv2 = Individu(pathMiddle1,pathMiddle2,path7,device=device)
        self.Indiv3 = Individu(pathRight1,pathRight2,path7,device=device)

        n1, n2, n3 = self.Indiv1.image_size, self.Indiv2.image_size, self.Indiv3.image_size

        self.Render1 = Rendu([n1,n1],device)
        self.Render2 = Rendu([n2,n2],device)
        self.Render3 = Rendu([n3,n3],device)
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
        
        #Hyperparamètres d'importance
        self.w1 = 1.
        self.w2 = 5.
        self.w3 = 1.

        #Hyperparamètres de la fonction de perte
        self.wlan = 1
        self.wc = 100.
        self.wregc = 2.5E-5
        self.wregl = 2.5E-7

        #Stockage des résultats de loss
        self.losslist = []


    def get_points(self, verts, Indiv, Render):              #projetter les points correspondant aux landmarks dans le plan de camera
        points = verts[:, self.Land_BFM, :]
        n = Indiv.image_size
        image_size = torch.tensor([[n,n]]).to(self.device)

        cameras = FoVPerspectiveCameras(device=self.device, R=Render.compute_R(), T=Render.T, fov = Render.fov)
        cordinates = cameras.transform_points_screen(points,image_size).to(self.device)
        return cordinates

    def loss_land(self, verts, Indiv, Render):
        points = self.get_points(verts, Indiv, Render) 
        points = points[:, :, 0:2]          # projected the points of the model
        distance = torch.sum((points - Indiv.landmarks) ** 2)         #calculate the distance
        return distance / 68

    def deformation(self,param_shape):
        deformation = torch.multiply(self.Face.PCA_basis, param_shape).reshape(self.n_shape, -1, 3).cuda()
        deformedface = self.Face.verts_mean + torch.sum(deformation, axis=0)
        return deformedface
    
    def loss_color(self, image, Indiv):
        distance = torch.sum((image[0,...,:3] - Indiv.image_contour) ** 2) 
        return distance / Indiv.card

    def loss_reg(self):
        S = 0
        S += torch.sum((self.Face.param_shape*self.Face.PCA_reg)**2)*self.wregl
        S += torch.sum((self.Face.param_color*self.Face.Color_reg)**2)*self.wregc
        return S

    def loss(self, image, Indiv, Render):
        S = 0
        S += self.wlan*self.loss_land(self.Face.verts, Indiv, Render)
        S += self.wc*self.loss_color(image, Indiv)
        S += self.loss_reg()
        return S

    def render_visual(self, index):
        if index ==1:
            image = self.Face.image1
            Indiv = self.Indiv1
            Render = self.Render1
        if index == 2:
            image = self.Face.image2
            Indiv = self.Indiv2
            Render = self.Render2
        if index == 3:
            image = self.Face.image3
            Indiv = self.Indiv3
            Render = self.Render3

        n = Indiv.image_size
        image_size = torch.tensor([[n,n]]).to(self.device) 
        plt.imshow(image[0,...,:3].cpu().detach().numpy())
        points = self.Face.verts[:, self.Land_BFM, :]

        cordinates = Render.cameras.transform_points_screen(points, image_size=image_size).to(self.device)
        plt.scatter(cordinates[0,...,0].cpu().detach().numpy(),cordinates[0,...,1].cpu().detach().numpy(), color='r')
        plt.scatter(Indiv.landmarks[0,...,0].cpu().detach().numpy(),Indiv.landmarks[0,...,1].cpu().detach().numpy())

        plt.show()

    def render(self, index):
        if index ==1:
            image = self.Face.image1
            plt.imshow(image[0, ..., :3].cpu().detach().numpy())
            plt.show()
        if index == 2:
            image = self.Face.image2
            plt.imshow(image[0, ..., :3].cpu().detach().numpy())
            plt.show()
        if index == 3:
            image = self.Face.image3
            plt.imshow(image[0, ..., :3].cpu().detach().numpy())
            plt.show()

    def supperposition(self, index):
        if index == 1:
            image = np.array(self.Face.image1[0,...,:3].detach().cpu())
            Indiv = self.Indiv1
        if index == 2:
            image = np.array(self.Face.image2[0,...,:3].detach().cpu())
            Indiv = self.Indiv2
        if index == 3:
            image = np.array(self.Face.image3[0,...,:3].detach().cpu())
            Indiv = self.Indiv3


        res = np.zeros(image.shape)
        ref = np.array(Indiv.image_ref.cpu())
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                if sum(image[i][j])==3.:
                    for index in range(3):
                        res[i][j][index] = ref[i][j][index]

                else: 
                    for index in range(3):
                        res[i][j][index] = image[i][j][index]
        return res



    def forward_individual(self, index, Indiv, Render):  
        #Calcul la forme et la couleur en fonction des paramètres 
        R = Render.compute_R()
        Render.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=Render.T, fov = Render.fov)
        self.Face.verts = self.deformation(self.Face.param_shape)
        self.Face.colors = self.Face.color_mean + (self.Face.param_color@self.Face.Color_basis).reshape(1,-1,3).to(self.device)
        self.Face.mesh = Meshes(verts=self.Face.verts, faces=self.Face.triangles,
                                textures=TexturesVertex(verts_features=self.Face.colors)).cuda()
        
        #Modifie les paramètres de rendu et stocke le rendu
        Render.lights = DirectionalLights(device = self.device, direction = Render.lights_direction)
        Render.renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=Render.cameras, 
                                                                 raster_settings=Render.raster_settings),
                                       shader=HardPhongShader(device=self.device, cameras=Render.cameras,
                                                              lights=Render.lights))
        n = Indiv.image_size
        image_size = torch.tensor([n,n]).cuda()
        loss = 0
        if index == 1:
            self.Face.image1 = Render.renderer(self.Face.mesh, materials=Render.materials,image_size=image_size).cuda()
            loss = self.loss(self.Face.image1, Indiv, Render)

        if index == 2:
            self.Face.image2 = Render.renderer(self.Face.mesh, materials=Render.materials,image_size=image_size).cuda()
            loss = self.loss(self.Face.image2, Indiv, Render)

        if index == 3:
            self.Face.image3 = Render.renderer(self.Face.mesh, materials=Render.materials,image_size=image_size).cuda()
            loss = self.loss(self.Face.image3, Indiv, Render)
        
        return loss

    def forward(self):
        loss1 = self.forward_individual(1, self.Indiv1, self.Render1)
        loss2 = self.forward_individual(2, self.Indiv2, self.Render2)
        loss3 = self.forward_individual(3, self.Indiv3, self.Render3)
        loss = self.w1*loss1 + self.w2*loss2 + self.w3*loss3
        
        return loss