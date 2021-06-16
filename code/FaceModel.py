import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import h5py
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras, FoVOrthographicCameras, PointLights, DirectionalLights 
from pytorch3d.renderer import Materials, RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader
from pytorch3d.renderer import TexturesVertex, Textures

class FaceModel(nn.Module):
    def __init__(self, n_shape, n_color, path='model2019_fullHead.h5', device='cuda:0'):
        f = h5py.File(path, 'r')
        self.device = device

        #Importation du modèle moyen
        self.Verts_mean = f['shape']['model']['mean'].value.reshape(1, -1)
        self.Color_mean = f['color']['model']['mean'].value.reshape(-1, 3)
        self.triangles = torch.from_numpy(f['shape']['representer']['cells'].value).T.unsqueeze(0).to(device)
        #Importation pour le modèle de la forme
        self.PCA_Var = np.array(f['shape']['model']['pcaVariance'])
        self.PCA_basis = torch.from_numpy(np.copy([np.transpose(
                                                                f['shape']['model']['pcaBasis']
                                                               )[0:n_shape,...]])).reshape(n_shape, -1).to(self.device)
        self.PCA_reg = self.pcaReg(self.PCA_Var, n_shape)


        #Importation pour le modèle de la couleur
        self.Color_Var = np.array(f['color']['model']['pcaVariance'])   
        self.Color_basis = torch.from_numpy(np.copy([np.transpose(
                                                                  f['color']['model']['pcaBasis']
                                                                 )[0:n_color,...]])).reshape(n_color, -1).to(self.device)
        self.Color_reg = self.pcaReg(self.Color_Var, n_color)
 
        #Création des paramètres
        super().__init__()
        self.param_color = nn.Parameter(torch.zeros(n_color).to(device))
        self.param_shape = nn.Parameter(torch.from_numpy(np.ones(n_shape, dtype=np.float32)).to(device).reshape(-1,1))
           
        #Création de la structure moyenne 
        self.verts_mean = torch.from_numpy(f['shape']['model']['mean'].value.reshape(1, -1, 3)).to(device)
        self.color_mean = torch.from_numpy(self.Color_mean).unsqueeze(0).to(device)+(self.param_color@self.Color_basis).reshape(1,-1,3)  
        self.textures=TexturesVertex(verts_features=self.color_mean.to(device))

        #Stockage du modèle de visage
        self.verts=torch.from_numpy(f['shape']['model']['mean'].value.reshape(1, -1, 3)).to(device)
        self.colors = self.color_mean
        self.mesh = Meshes(verts=self.verts, faces=self.triangles,
                           textures=TexturesVertex(verts_features=self.color_mean.to(self.device)))
        self.image = None

    def sample_face(self):
        #Génération de la déformation de la forme
        sample_s = []
        for i in range(80):
            sample_s.append(np.random.normal(0, np.sqrt(self.PCA_Var[i])))
            
        sample_s = torch.tensor(sample_s).to(self.device)
        #Génération de la déformation de la couleur
        sample_c = []
        for i in range(80):
            sample_c.append(np.random.normal(0, np.sqrt(self.Color_Var[i])))

        sample_c = torch.tensor(sample_c).to(self.device)
        
        verts_sample = (self.verts_mean.to(self.device)[0] + sample_s@self.PCA_basis).reshape(1, -1, 3)
        colors_sample = (torch.from_numpy(self.Color_mean).to(self.device).unsqueeze(0) + \
                         sample_c@self.Color_basis).reshape(1, -1, 3)

        mesh = Meshes(verts=verts_sample, faces=self.triangles,
                      textures=TexturesVertex(verts_features=colors_sample.to(self.device)))

        return mesh

    def mesh_mean(self):
        verts = torch.from_numpy(self.Verts_mean).to(self.device).reshape(1, -1, 3)
        colors = torch.from_numpy(self.Color_mean).to(self.device).unsqueeze(0).reshape(1, -1, 3)

        return Meshes(verts=verts, faces=self.triangles,
                      textures=TexturesVertex(verts_features=colors.to(self.device)))


    def pcaReg(self, PCA, size):
        L=[]
        for i in range(size):
            L.append(1. / PCA[i]**.5) #Tableau des inverses des écart-types du PCA
        return torch.tensor(L, device=self.device)