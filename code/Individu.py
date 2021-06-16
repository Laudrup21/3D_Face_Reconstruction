import numpy as np
import torch
from imutils import face_utils
import dlib
import cv2 as cv
from PIL import Image, ImageTk

class Individu():
    def __init__(self, path1="Left.png", path2=None, 
                 path3="shape_predictor_68_face_landmarks.dat", device="cuda"):

        #Ouverture des images
        image_avant = np.array(Image.open(path1), dtype=float) / 255
        image_contour = None
        
        if path2 == None: 
            image_contour = self.delete_background(path1)
            self.image_contour = torch.from_numpy(image_contour).to(device)
        else:
            image_contour = np.array(Image.open(path2), dtype=float)[...,:3] / 255
            self.image_contour = torch.from_numpy(image_contour).to(device)

        #Stockage des landmarks sous forme de tenseurs        
        self.landmarks = torch.tensor(self.find_landmarks(path1, path3)).to(device)
        self.image_ref = torch.from_numpy(image_avant).to(device)
        self.image_size = image_avant.shape[0]
        self.card = self.card_m(image_contour)
        

    def card_m(self, image):
        S = 0
        white = 3.
        for i in range(len(image)):
            for j in range(len(image[0])):
                if sum(image[i][j]) < white:
                    S += 1

        return S
    
    def find_landmarks(self, path1, path3):
        #Détection des landmarks     
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(path3)
        src = cv.imread(path1)
        image = cv.cvtColor(src, cv.COLOR_BGR2GRAY) 
        rects = detector(image, 0)
        # For each detected face, find the landmark.
        Landmarks = []
        for i, rect in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(image, rect)
            shape = face_utils.shape_to_np(shape)
            Landmarks.append(shape)
        return Landmarks
        
    def delete_background(self, path1):
        img = cv.imread(path1)
        img2 = np.array(Image.open(path1), dtype=float)[...,:3] / 255
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        n, m = img.shape[0], img.shape[1]
        size_n, size_m = int(0.9*n), int(0.9*m)
        rect = (int(0.1*n), int(0.1*n), size_n, size_m)
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(float)
        img2 = img2 * mask2[:, :, np.newaxis]
        
        for i in range(n):
            for j in range(m):
                if sum(img2[i][j]) == 0:
                    img2[i][j] = [1., 1., 1.]
                    
        return img2