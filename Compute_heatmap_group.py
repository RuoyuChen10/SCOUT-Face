# -*- coding: utf-8 -*-
"""
Created on 2021/1/21

@author: Ruoyu Chen
"""
import argparse
import os
import re

import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
import pickle
from PIL import Image
import tensorflow as tf
from numba import cuda as cuda_
from torchvision import transforms

from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation
from MTCNN_Portable.mtcnn import MTCNN

from tqdm import tqdm


def get_net(net_name, weight_path=None):
    """
    Get the model through the name
        net_name: Model name
        weight_path: Weight name
    return: model
    """
    if net_name in ['VGGFace2']:
        # load model
        from model.vggface_models.resnet import resnet50
        if weight_path is None:
            weight_path = "./checkpoint/resnet50_scratch_weight.pkl"
        net = resnet50(num_classes=8631)
        with open(weight_path, 'rb') as f:
            obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        net.load_state_dict(weights)
    elif net_name in ['partial_fc']:
        from model.partial_fc.iresnet import iresnet50
        net = iresnet50()
        if weight_path is None:
            weight_path = "./checkpoint/partial_fc_16backbone.pth"
        state_dict = torch.load(weight_path)
        net.load_state_dict(state_dict)
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    return net

def get_attribution_maps_methods(net, layer_name, method_name):
    '''
    Get_attribution_maps_methods
        method_name: include GradCAM, GradCAM++
    '''
    if method_name in ["GradCAM"]:
        attr_maps = GradCAM(net, layer_name)
    elif method_name in ["GradCAM++"]:
        attr_maps = GradCamPlusPlus(net, layer_name)
    return attr_maps

def get_last_conv_name(net):
    """
    Get the last convolutional layer name
        net: the model
    return: name of last layer.
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

def prepare_input(img, net_type):
    '''
    Prepare the input images, depend on different model
        img: the input data
        net_type: model name, depend on the name adopt different preprocessing method
    return: prepare data and the posion of img
    '''
    if net_type in ['VGGFace2']:
        shape=(224,224)
        mean_bgr = (131.0912, 103.8827, 91.4953)  # from resnet50_ft.prototxt
        im_shape = img.shape[:2]
        ratio = float(shape[0]) / np.min(im_shape)
        img = cv2.resize(
            img,
            dsize=(int(np.ceil(im_shape[1] * ratio)),   # width
                int(np.ceil(im_shape[0] * ratio)))  # height
        )
        new_shape = img.shape[:2]
        h_start = (new_shape[0] - shape[0])//2
        w_start = (new_shape[1] - shape[1])//2
        img = img[h_start:h_start+shape[0], w_start:w_start+shape[1]]
        img = img.astype(np.float32)-mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W       
    elif net_type in ['partial_fc']:
        shape=(112,112)
        TFS = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_shape = img.shape[:2]
        ratio = float(shape[0]) / np.min(im_shape)
        img = cv2.resize(
            img,
            dsize=(int(np.ceil(im_shape[1] * ratio)),   # width
                int(np.ceil(im_shape[0] * ratio)))  # height
        )
        new_shape = img.shape[:2]
        h_start = (new_shape[0] - shape[0])//2
        w_start = (new_shape[1] - shape[1])//2
        img = img[h_start:h_start+shape[0], w_start:w_start+shape[1]]
        img = TFS(np.array(img))
        img = img.cpu().data.numpy()
    return torch.tensor([img], requires_grad=True),h_start,w_start

def gen_cam(image, img_, mask, box, h_start, w_start):
    """
    Generate CAM
        image: Original Image
        img_ : Crop image
        mask: [H,W], computed from Grad-CAM
        box: Computed from MTCNN
        h_start and w_start: posion
    return: tuple(cam,heatmap)
    """
    shape = mask.shape
        #mask = cv2.resize(mask, shape)
    image = image.copy()
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    #heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    im_shape = img_.shape[:2]
    ratio = float(shape[0]) / np.min(im_shape)
    img_ = cv2.resize(
        img_,
        dsize=(int(np.ceil(im_shape[1] * ratio)),   # width
               int(np.ceil(im_shape[0] * ratio)))  # height
    )
    img_[h_start:h_start+shape[0], w_start:w_start+shape[1]] = 0.5*img_[h_start:h_start+shape[0], w_start:w_start+shape[1]] + 0.5*heatmap
    img_ = cv2.resize(
        img_,
        dsize=(int(np.ceil(im_shape[1])),   # width
               int(np.ceil(im_shape[0])))  # height
    )
    image[
        int(np.floor(box[1]-box[3]*0.15)):int(np.ceil(box[1]+box[3]*1.15)),
        int(np.floor(box[0]-box[2]*0.15)):int(np.ceil(box[0]+box[2]*1.15))] = img_
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    heatmap = heatmap[..., ::-1]  # gbr to rgb
    return image, heatmap.astype(np.uint8)

def norm_image(image):
    """
    Normalizaition images
        image: [H,W,C]
    return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_gb(grad):
    """
    guided back propagation Grad of input image
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.cpu().data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image, input_image_name, network, key, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)

def mkdir(name):
    '''
    make dir
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0

def load_img_and_box(img_path, detector):
    '''
    load input images and corresponding 5 landmarks
    '''
    #Reading image
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Detect 5 key point
    face = detector.detect_faces(img)[0]
    box = face["box"]
    return image, box

def box_crop(image, box):
    image = image[
        int(np.floor(box[1]-box[3]*0.15)):int(np.ceil(box[1]+box[3]*1.15)),
        int(np.floor(box[0]-box[2]*0.15)):int(np.ceil(box[0]+box[2]*1.15))]
    return image

def Descriminant_explanations(mask,mask_c,confidence_mask=None):
    '''
    discriminant for network
    '''
    discriminant = mask*(np.max(mask_c)-mask_c)
    if confidence_mask is not None:
        discriminant = discriminant * confidence_mask
    # Normalization
    discriminant -= np.min(discriminant)
    discriminant /= np.max(discriminant)
    return discriminant

def main(args):
    # mkdir the output dir
    mkdir(args.output_dir)
    mkdir(os.path.join(args.output_dir,"class1"))
    mkdir(os.path.join(args.output_dir,"class2"))

    # MTCNN Detector
    with tf.device('gpu:0'):
        detector = MTCNN()
        # Input
        img1,box1 = load_img_and_box(args.image_path_1, detector)
        img2,box2 = load_img_and_box(args.image_path_2, detector)
        img1_crop = box_crop(img1, box1)
        img2_crop = box_crop(img2, box2)
    
    # Clean the GPU memory of tensorflow
    cuda_.select_device(0)
    cuda_.close()

    # Network
    net = get_net(args.network, args.weight_path)
    
    # Grad-CAM
    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    attr_maps = get_attribution_maps_methods(net, layer_name, args.attribution_maps)
    
    # GuidedBackPropagation
    gbp = GuidedBackPropagation(net)

    # Prepare image
    inputs1,h_start1,w_start1 = prepare_input(img1_crop, args.network)
    inputs2,h_start2,w_start2 = prepare_input(img2_crop, args.network)
    
    # Output image
    image1_dict = {}
    image2_dict = {}

    # CAM
    mask1 = attr_maps(inputs1.cuda(), args.class_id_1)  # cam mask
    attr_maps.remove_handlers()
    mask1_c = attr_maps(inputs1.cuda(), args.class_id_2)  # cam mask
    attr_maps.remove_handlers()

    mask2 = attr_maps(inputs2.cuda(), args.class_id_2)  # cam mask
    attr_maps.remove_handlers()
    mask2_c = attr_maps(inputs2.cuda(), args.class_id_1)  # cam mask
    attr_maps.remove_handlers()

    # Descriminant explanations
    discriminant1 = Descriminant_explanations(mask1, mask1_c, mask1)
    discriminant2 = Descriminant_explanations(mask2, mask1_c, mask2)

    # Generate Guided Grad-CAM
    inputs1.grad.zero_()  # Zero gradient
    inputs2.grad.zero_()  # Zero gradient
    
    grad1 = gbp(inputs1.cpu())
    grad2 = gbp(inputs2.cpu())
    
    gb1 = gen_gb(grad1)
    gb2 = gen_gb(grad2)

    cam_gb1 = gb1 * mask1[..., np.newaxis]
    cam_gb2 = gb2 * mask1[..., np.newaxis]

    # Save image
    image1_dict[args.attribution_maps], image1_dict['heatmap'] = gen_cam(img1,img1_crop,mask1,box1,h_start1,w_start1)
    image1_dict[args.attribution_maps+'-id2'], image1_dict['heatmap-id2'] = gen_cam(img1,img1_crop,mask1_c,box1,h_start1,w_start1)
    image1_dict[args.attribution_maps+'-discriminant'], image1_dict['heatmap-discriminant'] = gen_cam(img1,img1_crop,discriminant1,box1,h_start1,w_start1)
    image1_dict['gb'] = norm_image(gb1)
    image1_dict['cam_gb'] = norm_image(cam_gb1)

    save_image(image1_dict[args.attribution_maps], os.path.basename(args.image_path_1), args.network, args.attribution_maps, os.path.join(args.output_dir,"class1"))
    save_image(image1_dict[args.attribution_maps+'-id2'], os.path.basename(args.image_path_1), args.network, args.attribution_maps+'-id2', os.path.join(args.output_dir,"class1"))
    save_image(image1_dict['heatmap'], os.path.basename(args.image_path_1), args.network, 'heatmap', os.path.join(args.output_dir,"class1"))
    save_image(image1_dict['heatmap-id2'], os.path.basename(args.image_path_1), args.network, 'heatmap-id2', os.path.join(args.output_dir,"class1"))
    save_image(image1_dict[args.attribution_maps+'-discriminant'], os.path.basename(args.image_path_1), args.network, args.attribution_maps+'-discriminant', os.path.join(args.output_dir,"class1"))
    save_image(image1_dict['heatmap-discriminant'], os.path.basename(args.image_path_1), args.network, 'heatmap-discriminant', os.path.join(args.output_dir,"class1"))
    save_image(image1_dict['gb'], os.path.basename(args.image_path_1), args.network, 'gb', os.path.join(args.output_dir,"class1"))
    save_image(image1_dict['cam_gb'], os.path.basename(args.image_path_1), args.network, 'cam_gb', os.path.join(args.output_dir,"class1"))

    image2_dict[args.attribution_maps], image2_dict['heatmap'] = gen_cam(img2,img2_crop,mask2,box2,h_start2,w_start2)
    image2_dict[args.attribution_maps+'-id2'], image2_dict['heatmap-id2'] = gen_cam(img2,img2_crop,mask2_c,box2,h_start2,w_start2)
    image2_dict[args.attribution_maps+'-discriminant'], image2_dict['heatmap-discriminant'] = gen_cam(img2,img2_crop,discriminant2,box2,h_start2,w_start2)
    image2_dict['gb'] = norm_image(gb2)
    image2_dict['cam_gb'] = norm_image(cam_gb2)

    save_image(image2_dict[args.attribution_maps], os.path.basename(args.image_path_2), args.network, args.attribution_maps, os.path.join(args.output_dir,"class2"))
    save_image(image2_dict[args.attribution_maps+'-id2'], os.path.basename(args.image_path_2), args.network, args.attribution_maps+'-id2', os.path.join(args.output_dir,"class2"))
    save_image(image2_dict['heatmap'], os.path.basename(args.image_path_2), args.network, 'heatmap', os.path.join(args.output_dir,"class2"))
    save_image(image2_dict['heatmap-id2'], os.path.basename(args.image_path_2), args.network, 'heatmap-id2', os.path.join(args.output_dir,"class2"))
    save_image(image2_dict[args.attribution_maps+'-discriminant'], os.path.basename(args.image_path_2), args.network, args.attribution_maps+'-discriminant', os.path.join(args.output_dir,"class2"))
    save_image(image2_dict['heatmap-discriminant'], os.path.basename(args.image_path_2), args.network, 'heatmap-discriminant', os.path.join(args.output_dir,"class2"))
    save_image(image2_dict['gb'], os.path.basename(args.image_path_2), args.network, 'gb', os.path.join(args.output_dir,"class2"))
    save_image(image2_dict['cam_gb'], os.path.basename(args.image_path_2), args.network, 'cam_gb', os.path.join(args.output_dir,"class2"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='VGGFace2',
                        help='ImageNet classification network')
    parser.add_argument('--image-path-1', type=str, default='./input/0001_01.jpg',
                        help='input image path')
    parser.add_argument('--image-path-2', type=str, default='./input/0012_01.jpg',
                        help='input renderer image path')
    parser.add_argument('--class-id-1', type=int, default=0,
                        help='class id')
    parser.add_argument('--class-id-2', type=int, default=1,
                        help='class id')
    parser.add_argument('--weight-path', type=str, default=None,
                        help='weight path of the model')
    parser.add_argument('--layer-name', type=str, default=None,
                        help='last convolutional layer name')
    parser.add_argument('--attribution-maps', type=str, default="GradCAM",
                        help='Method to generate heatmap, include GradCAM and GradCAM++')
    parser.add_argument('--output-dir', type=str, default='./results/',
                        help='output directory to save results')
    arguments = parser.parse_args()
    torch.backends.cudnn.enabled = True

    torch.backends.cudnn.benchmark = True
    main(arguments)
