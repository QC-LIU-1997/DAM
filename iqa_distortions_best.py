import numpy as np
from PIL import ImageFilter, ImageEnhance
import skimage.morphology
from scipy import ndimage
import random
from skimage import color,filters,io
from sklearn.preprocessing import normalize
import io
from scipy.interpolate import UnivariateSpline
import PIL
from scipy import interpolate
import skimage
from skimage.filters import gaussian
import math
import torchvision.transforms as T

from PIL.Image import Image
from torch import Tensor

from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#dist_level = [0,1,2,3,4]


def curvefit (xx, coef):


    x = np.array([0,0.5,1])
    y = np.array([0,coef,1])

    tck = UnivariateSpline(x, y, k=2)
    return np.clip(tck(xx),0,1)


def mapmm(e):

    mina = 0.0
    maxa = 1.0
    minx = np.min(e)
    maxx = np.max(e)
    if minx<maxx : 
        e = (e-minx)/(maxx-minx)*(maxa-mina)+mina
    return e



def Gradient(img):
    """
    Gradients of one image with symmetric boundary conditons
    
    Parameters
    -------
    img ； ndarray
    
    Returns
    ------
    grx : ndarry
        one-order froward  difference in the direction of column(axis = 1)
    gry : ndarry
        one-order froward  difference in the direction of row   (axis = 0)
    glx : ndarry
        one-order backward difference in the direction of column(axis = 1)
    gly : ndarry
        one-order backward difference in the direction of row   (axis = 0)
    grc : ndarry
        self-defined difference function    
    """
    #img(i,j-1)
    img_right = np.roll(img,1,axis = 1)
    img_right[:,0] = img[:,0]
    #img(i,j+1)
    img_left  = np.roll(img,-1,axis = 1)
    img_left[:,-1] = img[:,-1]
    #img(i+1,j)
    img_up = np.roll(img,-1,axis = 0)
    img_up[-1] = img[-1]
    #img(i-1,j)
    img_down = np.roll(img,1,axis = 0)
    img_down[0] = img[0]
    
    #img(i,j+1) - img(i,j)
    grx = img_left - img 
    #img(i+1,j) - img(i,j)
    gry = img_up - img
    #img(i,j)  - img(i,j-1)
    glx = img - img_right 
    #img(i,j)   - img(i-1,j)
    gly = img - img_down   
    #img(i,j+1) + img(i+1,j)+ img(i,j-1) +img(i-1,j)  - 4*I(i,j)
    grc = grx+gry-glx-gly  
    return grx,gry,glx,gly,grc

def qq(img):
    """
    Instantaneous coefficient of variation: q(x,y,t)
    
    Parameters
    ------
    img: ndarray
    
    Returns
    ------
    q : ndarray
        The formula is as follows:
        q(x, y ; t)=\sqrt{\frac{(1 / 2)(|\nabla I| / I)^{2}
        -\left(1 / 4^{2}\right)\left(\nabla^{2} I / I\right)^{2}}
        {\left[1+(1 / 4)\left(\nabla^{2} I / I\right)\right]^{2}}}
    """
    grx,gry,glx,gly,grc = Gradient(img)
    q_1 = (grx**2+gry**2+glx**2+gly**2)**0.5/(img+1e-06)
    q_2 = grc /(img+1e-06)
    q   = ((1/2*q_1**2 - 1/16*q_2**2) / ((1+1/4*q_2)**2)+1e-06)**0.5
    
    return q
    
def imblursrad(img,level):
    # levels = [0.04, 0.06, 0.065, 0.07, 0.75, 0.08, 0.085, 0.09]
    levels = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    delta_t = levels[level]
    k = 30
    m = 10
    q_0 = 1
    rho = 1
    Iterations = 1

    img = np.array(img)


    """
    speckle reducing anistropic diffusion
    
    Parameter
    ------
    img: ndarray
    
    k:  number
        attenuation coefficient
    m； number
        control rate of homogeneous area
    q_0: number
        the threshold of intial speckle noise
    rho: number
    delta_t:number
        timespace
    Iteration: number
        the number of iterations
    
    Returns
    img: ndarray 
        the image after being filtered by srad
    """

    img = img/1.0
    for i in range(0,Iterations):
        grx,gry,glx,gly,grc = Gradient(img)
    
        # compute the diffusion coefficient
        q  = qq(img)
        q_t = q_0*math.exp(-rho*i*delta_t)
        cq = np.pi/2 - np.arctan(k*(q**2 - m*q_t**2))
        
        # cq(i+1,j)
        cq_up = np.roll(cq,-1,axis = 0)
        cq_up[-1] = cq[-1]
        # cq(i,j+1)
        cq_left = np.roll(cq,-1,axis = 1)
        cq_left[:,-1] = cq[:,-1]
        
        Div = cq_up*gry - cq*gly + cq_left*grx-cq*glx
        img_out = np.clip(img + 1/4*delta_t*Div,0,255)
    return PIL.Image.fromarray(img_out)

def imblurgauss(im, level):
    # Takes in PIL Image and returns Gaussian Blurred PIL Image
    levels = [0.4, 0.44, 0.48, 0.52, 0.56, 0.6]
    sigma = levels[level]
    
    im = np.array(im)

    im_dist = cv2.GaussianBlur(im/1.0,ksize = (7,7), sigmaX=sigma)
    im_dist = np.clip(im_dist,0,255)
    return PIL.Image.fromarray(im_dist)

def imbrighten(im,level):
    # levels = [0.002, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03]
    levels = [0.01, 0.015, 0.02, 0.025, 0.03,0.035]
    im = np.array(im)

    adj = 1.0-levels[level]
    im_out = (im)**adj
    return PIL.Image.fromarray(im_out)

def imdarken(im,level):
    # levels = [0.002, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03]
    levels = [0.01, 0.015, 0.02, 0.025, 0.03,0.035]
    im = np.array(im)
    adj = 1.0+levels[level]
    im_out = (im)**adj
    return PIL.Image.fromarray(im_out)


def imp_average_pooling(img, G=4):
 
    out = img.copy()
    H, W = img.shape
    Nh = int(H / G)
    Nw = int(W / G)
    for y in range(Nh):
        for x in range(Nw):
            out[G*y:G*(y+1), G*x:G*(x+1)] = np.mean(out[G*y:G*(y+1), G*x:G*(x+1)])
    return out

def imp_multiplicative_noise(im):

    # levels = [0.001, 0.005, 0.01, 0.02, 0.05]

    im = im/255.0

    row,col= im.shape

    var = 0.2
    mean = 0
    sigma = var**0.5
    gauss = np.array(im.shape)
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = im + im * gauss
    noisy = noisy * 255.0
    noisy = np.clip(noisy,0,255)

    return noisy

def imcutblur(im, mask, level):
    rect_flag = random.randint(0,1)
    levels = [(4,8),(8,8),(8,16),(16,16),(16,32),(32,32)]
    # levels = [(24,24),(24,48),(48,48),(48,96),(96,96),(96,192),(192,192)]
    param = levels[level]

    ind = round(np.max(param)/2)
    im = np.array(im)
    mask = np.array(mask)
    mask_ind = np.zeros_like(mask)
    mask_ind[ind:-ind,ind:-ind] = 1
    mask_ind = mask*mask_ind
    idx_eff = np.where(mask_ind>0)
    target_idx_cand = random.randint(0,len(idx_eff[0])-1)
    if rect_flag == 0:

        target_y_start = idx_eff[0][target_idx_cand]-round(param[0]/2)
        target_y_end = idx_eff[0][target_idx_cand]+round(param[0]/2)
        target_x_start = idx_eff[1][target_idx_cand]-round(param[1]/2)
        target_x_end = idx_eff[1][target_idx_cand]+round(param[1]/2)

    else:

        target_y_start = idx_eff[0][target_idx_cand]-round(param[1]/2)
        target_y_end = idx_eff[0][target_idx_cand]+round(param[1]/2)
        target_x_start = idx_eff[1][target_idx_cand]-round(param[0]/2)
        target_x_end = idx_eff[1][target_idx_cand]+round(param[0]/2)



    p_change = im[target_y_start:target_y_end,target_x_start:target_x_end]
    p_change = (mask[target_y_start:target_y_end,target_x_start:target_x_end]/255.0)*imp_average_pooling(p_change)
    p_hold = (1-mask[target_y_start:target_y_end,target_x_start:target_x_end]/255.0)*im[target_y_start:target_y_end,target_x_start:target_x_end]
    p_fin = p_change+p_hold

    im[target_y_start:target_y_end,target_x_start:target_x_end] = p_fin


    im_out = PIL.Image.fromarray(im)
    return im_out

def imcutpaste(im, mask, level):
    rect_flag = random.randint(0,1)
    levels = [(4,8),(8,8),(8,16),(16,16),(16,32),(32,32)]
    # [(4,4),(4,8),(8,8),(8,16),(16,16),(16,16),(16,32),(32,32),(32,64),(64,64),(64,128),(128,128)]
    # levels = [(24,24),(24,48),(48,48),(48,96),(96,96),(96,192),(192,192)]
    param = levels[level]

    ind = round(np.max(param)/2)
    im = np.array(im)
    mask = np.array(mask)
    mask_ind = np.zeros_like(mask)
    mask_ind[ind:-ind,ind:-ind] = 1
    mask_ind = mask*mask_ind
    idx_eff = np.where(mask_ind>0)
    source_idx_cand = random.randint(0,len(idx_eff[0])-1)
    target_idx_cand = random.randint(0,len(idx_eff[0])-1)
    if rect_flag == 0:

        source_y_start = idx_eff[0][source_idx_cand]-round(param[0]/2)
        source_y_end = idx_eff[0][source_idx_cand]+round(param[0]/2)
        source_x_start = idx_eff[1][source_idx_cand]-round(param[1]/2)
        source_x_end = idx_eff[1][source_idx_cand]+round(param[1]/2)

        target_y_start = idx_eff[0][target_idx_cand]-round(param[0]/2)
        target_y_end = idx_eff[0][target_idx_cand]+round(param[0]/2)
        target_x_start = idx_eff[1][target_idx_cand]-round(param[1]/2)
        target_x_end = idx_eff[1][target_idx_cand]+round(param[1]/2)

    else:

        source_y_start = idx_eff[0][source_idx_cand]-round(param[1]/2)
        source_y_end = idx_eff[0][source_idx_cand]+round(param[1]/2)
        source_x_start = idx_eff[1][source_idx_cand]-round(param[0]/2)
        source_x_end = idx_eff[1][source_idx_cand]+round(param[0]/2)

        target_y_start = idx_eff[0][target_idx_cand]-round(param[1]/2)
        target_y_end = idx_eff[0][target_idx_cand]+round(param[1]/2)
        target_x_start = idx_eff[1][target_idx_cand]-round(param[0]/2)
        target_x_end = idx_eff[1][target_idx_cand]+round(param[0]/2)



    p_change = (mask[source_y_start:source_y_end,source_x_start:source_x_end]/255.0)*(mask[target_y_start:target_y_end,target_x_start:target_x_end]/255.0)*im[source_y_start:source_y_end,source_x_start:source_x_end]
    p_hold = (1-(mask[source_y_start:source_y_end,source_x_start:source_x_end]/255.0)*(mask[target_y_start:target_y_end,target_x_start:target_x_end]/255.0))*im[target_y_start:target_y_end,target_x_start:target_x_end]
    p_fin = p_change+p_hold

    im[target_y_start:target_y_end,target_x_start:target_x_end] = p_fin


    im_out = PIL.Image.fromarray(im)
    return im_out


def imcutnoise(im, mask, level):
    rect_flag = random.randint(0,1)
    levels = [(4,8),(8,8),(8,16),(16,16),(16,32),(32,32)]
    # levels = [(24,24),(24,48),(48,48),(48,96),(96,96),(96,192),(192,192)]
    param = levels[level]

    ind = round(np.max(param)/2)
    im = np.array(im)
    mask = np.array(mask)
    mask_ind = np.zeros_like(mask)
    mask_ind[ind:-ind,ind:-ind] = 1
    mask_ind = mask*mask_ind
    idx_eff = np.where(mask_ind>0)
    target_idx_cand = random.randint(0,len(idx_eff[0])-1)
    if rect_flag == 0:

        target_y_start = idx_eff[0][target_idx_cand]-round(param[0]/2)
        target_y_end = idx_eff[0][target_idx_cand]+round(param[0]/2)
        target_x_start = idx_eff[1][target_idx_cand]-round(param[1]/2)
        target_x_end = idx_eff[1][target_idx_cand]+round(param[1]/2)

    else:

        target_y_start = idx_eff[0][target_idx_cand]-round(param[1]/2)
        target_y_end = idx_eff[0][target_idx_cand]+round(param[1]/2)
        target_x_start = idx_eff[1][target_idx_cand]-round(param[0]/2)
        target_x_end = idx_eff[1][target_idx_cand]+round(param[0]/2)



    p_change = im[target_y_start:target_y_end,target_x_start:target_x_end]
    p_change = (mask[target_y_start:target_y_end,target_x_start:target_x_end]/255.0)*imp_multiplicative_noise(p_change)
    p_hold = (1-mask[target_y_start:target_y_end,target_x_start:target_x_end]/255.0)*im[target_y_start:target_y_end,target_x_start:target_x_end]
    p_fin = p_change+p_hold

    im[target_y_start:target_y_end,target_x_start:target_x_end] = p_fin


    im_out = PIL.Image.fromarray(im)
    return im_out


def pre_shadow_gene(param):
    full_mask = np.ones((param[0]*2,param[1]*2))/1.0
    full_mask[param[0]-param[0]//2:param[0]+param[0]//2,param[1]-param[1]//2:param[1]+param[1]//2]=0.0
    ksize = np.min(param)//2-1
    full_mask = cv2.GaussianBlur(full_mask, (ksize, ksize),0, borderType = 1)
    cut_mask = full_mask[param[0]//4:-param[0]//4,param[1]//4:-param[1]//4]
    return cut_mask


# pre_shadow_list = [pre_shadow_gene((16,16)),pre_shadow_gene((16,32)),pre_shadow_gene((32,32)),pre_shadow_gene((32,64)),pre_shadow_gene((64,64)),pre_shadow_gene((64,128)),pre_shadow_gene((128,128))]
pre_shadow_list = [pre_shadow_gene((4,8)),pre_shadow_gene((8,8)),pre_shadow_gene((8,16)),pre_shadow_gene((16,16)),pre_shadow_gene((16,32)),pre_shadow_gene((32,32))]

def imcutshadow(im, mask, level):
    rect_flag = random.randint(0,1)
    levels = [(4,8),(8,8),(8,16),(16,16),(16,32),(32,32)]
    # levels = [(24,24),(24,48),(48,48),(48,96),(96,96),(96,192),(192,192)]
    param = levels[level]
    pre_shadow = pre_shadow_list[level]

    ind = round(np.max(param)/2)
    im = np.array(im)
    mask = np.array(mask)
    mask_ind = np.zeros_like(mask)
    mask_ind[ind:-ind,ind:-ind] = 1
    mask_ind = mask*mask_ind
    idx_eff = np.where(mask_ind>0)
    target_idx_cand = random.randint(0,len(idx_eff[0])-1)

    if rect_flag == 0:
        target_y_start = idx_eff[0][target_idx_cand]-round(param[0]/2)
        target_y_end = idx_eff[0][target_idx_cand]+round(param[0]/2)
        target_x_start = idx_eff[1][target_idx_cand]-round(param[1]/2)
        target_x_end = idx_eff[1][target_idx_cand]+round(param[1]/2)

    else:
        target_y_start = idx_eff[0][target_idx_cand]-round(param[1]/2)
        target_y_end = idx_eff[0][target_idx_cand]+round(param[1]/2)
        target_x_start = idx_eff[1][target_idx_cand]-round(param[0]/2)
        target_x_end = idx_eff[1][target_idx_cand]+round(param[0]/2)
        pre_shadow = pre_shadow.T


    shadow = np.ones_like(im)/1.0

    p_y_change = target_y_end-target_y_start
    p_x_change = target_x_end-target_x_start

    cut_y_start = max(0,target_y_start-p_y_change//4)
    cut_y_end = min(im.shape[0],target_y_end+p_y_change//4)
    cut_x_start = max(0,target_x_start-p_x_change//4)
    cut_x_end = min(im.shape[1],target_x_end+p_x_change//4)
    paste_y_start = pre_shadow.shape[0]//2-(idx_eff[0][target_idx_cand]-cut_y_start)
    paste_y_end = pre_shadow.shape[0]//2+(cut_y_end-idx_eff[0][target_idx_cand])    
    paste_x_start = pre_shadow.shape[1]//2-(idx_eff[1][target_idx_cand]-cut_x_start)
    paste_x_end = pre_shadow.shape[1]//2+(cut_x_end-idx_eff[1][target_idx_cand])
    shadow[cut_y_start:cut_y_end,cut_x_start:cut_x_end] = pre_shadow[paste_y_start:paste_y_end,paste_x_start:paste_x_end]
    im = (im*shadow)
    # print(type(im[0][0]),type(shadow[0][0]))
    # print(im.min(),im.max(),shadow.min(),shadow.max())


    im_out = PIL.Image.fromarray(im)
    return im_out

def imglobalpass(im, level):
    return im
def imlocalpass(im, mask, level):
    return im

"""
im = Image.open("IMG_1651.png")
for level in dist_level:
    
    im_dist = imcompressjpeg(im,level)
    im_dist.save("level"+str(level)+".png")

"""

class ViewTransform_Nonde:
    def __init__(
        self,
        crop_size: int = 224,
        hf_prob: float = 0.5,
        per_prob: float = 0.3,
        distortion_scale: float = 0.3,
        theta: float = 15,
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
    ):
        transform = [
            T.Resize(crop_size),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomRotation(theta,expand=True),
            T.RandomPerspective(distortion_scale=0.3,p=per_prob),
            T.RandomAffine(degrees=0, shear=[-5,5,-5,5]),
            T.CenterCrop(crop_size),
            T.ToTensor(),T.Normalize((0.5), (0.5))
        ]
        self.transform = T.Compose(transform)
    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """

        _,mask,im = image.convert('RGB').split()
        im = PIL.Image.fromarray(np.array(im)/1.0)

        transformed: Tensor = self.transform(im)
        return transformed

total_global = ['imblurgauss','imblursrad','imbrighten','imdarken']
total_local = ['imcutblur','imcutpaste','imcutnoise','imcutshadow']


class ViewTransform_De:
    def __init__(
        self,
        crop_size: int = 224,
        hf_prob: float = 0.5,
        per_prob: float = 0.3,
        distortion_scale: float = 0.3,
        theta: float = 15,
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        level = 6,
        detype = 'comp',
        mc_flag = False
    ):
        transform = [
            T.Resize(crop_size),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomRotation(theta,expand=True),
            T.RandomPerspective(distortion_scale=0.3,p=per_prob),
            T.RandomAffine(degrees=0, shear=[-5,5,-5,5]),
            T.CenterCrop(crop_size),
            T.ToTensor(),T.Normalize((0.5), (0.5))
        ]
        self.transform = T.Compose(transform)
        self.detype = detype
        self.level = level
    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        _,mask,im = image.convert('RGB').split()
        im = np.array(im)/1.0
        global_sele = random.sample(total_global, 1)[0]
        local_sele = random.sample(total_local, 1)[0]
        if self.detype == 'comp':
            im = eval(local_sele)(im,mask,self.level)
            im = eval(global_sele)(im,self.level)
        elif self.detype == 'alt':
            chosen_flag = random.randint(0,1)
            if chosen_flag == 0:
                im = eval(local_sele)(im,mask,self.level)
            else:
                im = eval(global_sele)(im,self.level)
        elif self.detype == 'glo':
            im = eval(global_sele)(im,self.level)
        elif self.detype == 'loc':
            im = eval(local_sele)(im,mask,self.level)
        elif self.detype == 'imblurgauss':
            im = imblurgauss(im,self.level)
        elif self.detype == 'imblursrad':
            im = imblursrad(im,self.level)
        elif self.detype == 'imbrighten':
            im = imbrighten(im,self.level)
        elif self.detype == 'imdarken':
            im = imdarken(im,self.level)
        elif self.detype == 'imcutblur':
            im = imcutblur(im,mask,self.level)
        elif self.detype == 'imcutpaste':
            im = imcutpaste(im,mask,self.level)
        elif self.detype == 'imcutnoise':
            im = imcutnoise(im,mask,self.level)
        elif self.detype == 'imcutshadow':
            im = imcutshadow(im,mask,self.level)
        else:
            pass


        transformed: Tensor = self.transform(im)
        return transformed

class MultiViewTransform:
    """Transforms an image into multiple views.

    Args:
        transforms:
            A sequence of transforms. Every transform creates a new view.

    """

    def __init__(self, transforms: Sequence[T.Compose]):
        self.transforms = transforms

    def __call__(self, image: Union[Tensor, Image]) -> Union[List[Tensor], List[Image]]:
        """Transforms an image into multiple views.

        Every transform in self.transforms creates a new view.

        Args:
            image:
                Image to be transformed into multiple views.

        Returns:
            List of views.

        """
        return [transform(image) for transform in self.transforms]


class Transform_Global_WD(MultiViewTransform):

    def __init__(
        self,
        crop_size: int = 224,
        hf_prob: float = 0.5,
        per_prob: float = 0.3,
        distortion_scale: float = 0.3,
        theta: float = 15,
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        level = 6,
        detype = 'comp',
        scale_list = [],
        point_list = []
    ):
        # first global crop
        global_transform_0 = ViewTransform_Nonde(
            crop_size=crop_size,
            hf_prob=hf_prob,
            per_prob=per_prob,
            distortion_scale = distortion_scale,
            theta = theta,
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
        )

        # second global crop
        global_transform_1 = ViewTransform_Nonde(
            crop_size=crop_size,
            hf_prob=hf_prob,
            per_prob=per_prob,
            distortion_scale = distortion_scale,
            theta = theta,
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
        )

        global_transform_2 = ViewTransform_De(
            crop_size=crop_size,
            hf_prob=hf_prob,
            per_prob=per_prob,
            distortion_scale = distortion_scale,
            theta = theta,
            kernel_size=kernel_size,
            kernel_scale=kernel_scale,
            level = level,
            detype = detype
        )
        transforms = [global_transform_0, global_transform_1, global_transform_2]
        super().__init__(transforms)

class Transform_Multi_WD(MultiViewTransform):

    def __init__(
        self,
        crop_size: int = 224,
        hf_prob: float = 0.5,
        per_prob: float = 0.3,
        distortion_scale: float = 0.3,
        theta: float = 15,
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        level = 6,
        detype = 'comp',
        scalesize_list = [280,224,168],
    ):
        transforms = []
        # first global crop
        for trans_i in range(len(scalesize_list)):
            global_transform_0 = ViewTransform_Nonde(
                crop_size=scalesize_list[trans_i],
                hf_prob=hf_prob,
                per_prob=per_prob,
                distortion_scale = distortion_scale,
                theta = theta,
                kernel_size=kernel_size,
                kernel_scale=kernel_scale,
            )
            transforms.append(global_transform_0)

        for trans_i in range(len(scalesize_list)):
            # second global crop
            global_transform_1 = ViewTransform_Nonde(
                crop_size=scalesize_list[trans_i],
                hf_prob=hf_prob,
                per_prob=per_prob,
                distortion_scale = distortion_scale,
                theta = theta,
                kernel_size=kernel_size,
                kernel_scale=kernel_scale,
            )
            transforms.append(global_transform_1)

        for trans_i in range(len(scalesize_list)):
            global_transform_2 = ViewTransform_De(
                crop_size=scalesize_list[trans_i],
                hf_prob=hf_prob,
                per_prob=per_prob,
                distortion_scale = distortion_scale,
                theta = theta,
                kernel_size=kernel_size,
                kernel_scale=kernel_scale,
                level = level,
                detype = detype
            )
            transforms.append(global_transform_2)
        super().__init__(transforms)