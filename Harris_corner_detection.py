import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

def extract_keypoints(image):
        #setting the constant that will be used for calculating cornerness score
        k = 0.05
        
        #window size being considered
        window_size = 5
        
        #reading the image as a grayscale image
        if(type(image) == str):
                image_gray = cv2.imread(image, 0)
        else:
                #This is for part7 where the input parameter is 100x100 black and white image
                image_gray = image

        Ix = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0)
        Iy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1)
        
        #corness score will be stored in this image
        R = np.zeros(image_gray.shape)

        #the range starts from 2 so that R values withing 2 at top/left and bottom/right is kept as 0
        for r in range(2,image_gray.shape[0]-2):
                for c in range(2, image_gray.shape[1]-2):
                        #for each candidate point M is initialized to 2x2 zero matrix
                        M = np.zeros((2,2))
                        for i in range(r - int(window_size/2), r + int(window_size/2) + 1):
                                for j in range(c - int(window_size/2), c + int(window_size/2) + 1):
                                        if i==0 and j==0:
                                                continue
                                        #sum up all the Ix and Iy values in the matrix M
                                        M[0,0] += Ix[i,j] * Ix[i,j]
                                        M[0,1] += Ix[i,j]*Iy[i,j]
                                        M[1,0] += Ix[i,j]*Iy[i,j]
                                        M[1,1] += Iy[i,j]*Iy[i,j]
                        #computing lambda values using by singular value decomposition
                        u,s,v = np.linalg.svd(M)
                        [lmda1, lmda2] = s

                        #computing the R value for (r,c) using lambda values
                        lambda_product = lmda1*lmda2
                        lambda_sum = lmda1 + lmda2
                        R[r,c] = lambda_product - k*(lambda_sum**2)

        #Considering only those points which are less than the 5 times of the threshold
        R = (R>5*abs(np.mean(R)))*R

        #Non maximum suppression
        for r in range(1,image_gray.shape[0]-1):
                for c in range(1, image_gray.shape[1]-1):
                        flag = 0
                        for i in [r-1,r+1]:
                                for j in [c-1,c+1]:
                                        if(R[r,c] < R[i,j]):
                                                R[r,c] = 0
                                                flag = 1
                                                break
                                if(flag == 1):
                                        break


        (X, Y) = np.where(R != 0)
        return [X, Y, R[X,Y]]


if __name__ == "__main__":
    image = 'panda1.jpg'
    [X, Y, R] = extract_keypoints(image)
    R = (R/np.max(R))*100
    im = cv2.imread(image, 0)
    fig, ax  = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(im)
    for i in range(len(X)):
            ax.add_patch(mpl.patches.Circle((Y[i], X[i]), R[i], color='r', linewidth=0.3, fill= False))
    plt.show()
