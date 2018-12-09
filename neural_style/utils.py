import torch as th
import torch.nn as nn
import torchvision.transforms as tn
import numpy as np
from PIL import Image


# Preprocess an image before passing it to a model.
# We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
# and subtract the mean pixel.
def preprocess(image_name, image_size):
    if isinstance(image_name, str):
        image = Image.open(image_name).convert('RGB')
        if type(image_size) is not tuple:
            image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    elif isinstance(image_name, Image):
        image = image_name
    else:
        print('Please specifiy an image or a path to an image')
    loader = tn.Compose([tn.Resize(image_size), tn.ToTensor()])
    norm = tn.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])
    rgb2bgr = tn.Lambda(lambda x: x[th.LongTensor([2,1,0])])
    image = loader(image)
    tensor = norm(rgb2bgr(image * 255))
    return tensor.unsqueeze(0)
 

#  Undo the above preprocessing.
def deprocess(output_tensor):
    bgr2rgb = tn.Lambda(lambda x: x[th.LongTensor([2,1,0])])
    norm = tn.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1])
    output_tensor = norm(bgr2rgb(output_tensor.detach().clone().squeeze(0)))/255
    image = tn.ToPILImage()(output_tensor.clamp(0, 1).cpu())
    return image


# Combine the Y channel of the generated image and the UV/CbCr channels of the
# content image to perform color-independent style transfer.
def original_colors(content, generated):
    content_channels = list(content.convert('YCbCr').split())
    generated_channels = list(generated.convert('YCbCr').split())
    content_channels[0] = generated_channels[0]
    return Image.merge('YCbCr', content_channels).convert('RGB')


def match_color(target_img, source_img, mode='pca', eps=1e-5):
    '''
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be tensors [0, 1] and are returned as such.
    Modes are chol, pca or sym for different choices of basis.
    '''
    bgr2rgb = tn.Lambda(lambda x: x[th.LongTensor([2,1,0])])
    denorm = tn.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1])
    target_img = denorm(bgr2rgb(target_img.detach().clone().squeeze(0))) / 255
    target_img = target_img.permute(1,2,0).cpu().numpy()
    source_img = denorm(bgr2rgb(source_img.detach().clone().squeeze(0))) / 255
    source_img = source_img.permute(1,2,0).cpu().numpy()
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2,0,1).reshape(3,-1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2,0,1).reshape(3,-1)
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
    matched_img = ts.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
    matched_img = matched_img + mu_s
    matched_img[matched_img>1] = 1
    matched_img[matched_img<0] = 0
    matched_img = matched_img * 255
    norm = tn.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])
    tensor = norm(bgr2rgb(tn.ToTensor()(matched_img))).unsqueeze(0)
    return tensor