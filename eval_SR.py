import os
import lpips
import numpy as np
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from imageio.v2 import imread
import skimage
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import rgb2lab
import scipy
import cv2
import os

# set cuda to device 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # vgg is used in the paper
# loss_fn_vgg = lpips.LPIPS(net='alex').cuda()


def load_item(gt_path, pre_path, mask_path):


    gt = imread(gt_path)
    try:
        pre = imread(pre_path)
    except:
        pre = imread(pre_path.replace('.JPG', '.png'))
    if mask_path is not None:
        mask = imread(mask_path)


    # resize to gt size
    pre = resize(pre, (gt.shape[0], gt.shape[1]))
    if mask_path is not None:
        mask = resize(mask, (gt.shape[0], gt.shape[1]))
        # resize to pred size
        # pre = resize(pre, (512, 512))
        # mask = resize(mask, (512, 512))

        mask = (mask > 255 * 0.9).astype(np.uint8) * 255

    if mask_path is not None:
        return to_tensor(gt), to_tensor(pre), to_tensor(mask)
    else:
        return to_tensor(gt), to_tensor(pre), None


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    img_t = img_t.unsqueeze(dim=0)
    return img_t


def resize(img, target_size):
    img = skimage.transform.resize(img, target_size, mode='reflect', anti_aliasing=True)
    img = (img * 255).astype(np.uint8)  # Ensure the image is in uint8 format

    return img

def calc_rmse(real_img, fake_img):
    # Convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    rmse = np.sqrt(((real_lab - fake_lab) ** 2).mean())
    return rmse


def metric(gt, pre):
    transf = torchvision.transforms.Compose(
                [torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                
    lpips_value = loss_fn_vgg(transf(pre[0]).cuda(), transf(gt[0]).cuda()).item()

    pre = pre * 255.0
    pre = pre.permute(0, 2, 3, 1)
    pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

    gt = gt * 255.0
    gt = gt.permute(0, 2, 3, 1)
    gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

    psnr = compare_psnr(gt, pre)
    ssim = compare_ssim(gt, pre, data_range=255, channel_axis=-1)
    rmse = calc_rmse(gt, pre)

    return psnr, ssim, lpips_value, rmse



def evaluation(gt_root, pre_root, mask_root):
    fnames = os.listdir(gt_root)
    fnames.sort()

    psnr_all_list, ssim_all_list, lpips_all_list, rmse_all_list = [], [], [], []
    psnr_non_list, ssim_non_list, lpips_non_list, rmse_non_list = [], [], [], []
    psnr_shadow_list, ssim_shadow_list, lpips_shadow_list, rmse_shadow_list = [], [], [], []

    for fname in fnames:
        gt_path = os.path.join(gt_root, fname)
        pre_path = os.path.join(pre_root, fname)
        if mask_root is not None:
            mask_path = os.path.join(mask_root, fname)

        # For SDR only, replace the mask path _free.jpg to .png
        if mask_root is not None:
            mask_path = mask_path.replace('.jpg', '.png')
        else:
            mask_path = None
        pre_path = pre_path.replace('.jpg', '.png')
        if not os.path.exists(pre_path):
            pre_path = pre_path.replace('.png', '.jpg')


        gt, pre, mask = load_item(gt_path, pre_path, mask_path)

        psnr_all, ssim_all, lpips_all, rmse_all = metric(gt, pre)
        # psnr_non, ssim_non, lpips_non, rmse_non = metric(gt * (1 - mask), pre * (1 - mask))
        # psnr_shadow, ssim_shadow, lpips_shadow, rmse_shadow = metric(gt * mask, pre * mask)

        psnr_all_list.append(psnr_all)
        ssim_all_list.append(ssim_all)
        lpips_all_list.append(lpips_all)
        rmse_all_list.append(rmse_all)

        # psnr_non_list.append(psnr_non)
        # ssim_non_list.append(ssim_non)
        # lpips_non_list.append(lpips_non)
        # rmse_non_list.append(rmse_non)

        # psnr_shadow_list.append(psnr_shadow)
        # ssim_shadow_list.append(ssim_shadow)
        # lpips_shadow_list.append(lpips_shadow)
        # rmse_shadow_list.append(rmse_shadow)

        # print(f'ALL psnr: {round(psnr_all, 4)}/{round(np.average(psnr_all_list), 4)}  '
        #       f'ssim: {round(ssim_all, 4)}/{round(np.average(ssim_all_list), 4)}  '
        #       f'lpips: {round(lpips_all, 4)}/{round(np.average(lpips_all_list), 4)}  '
        #       f'rmse: {round(rmse_all, 4)}/{round(np.average(rmse_all_list), 4)} | '

        #       f'Shadow psnr: {round(psnr_shadow, 4)}/{round(np.average(psnr_shadow_list), 4)}  '
        #       f'ssim: {round(ssim_shadow, 4)}/{round(np.average(ssim_shadow_list), 4)}  '
        #       f'lpips: {round(lpips_shadow, 4)}/{round(np.average(lpips_shadow_list), 4)}  '
        #       f'rmse: {round(rmse_shadow, 4)}/{round(np.average(rmse_shadow_list), 4)} | '

        #       f'Non psnr: {round(psnr_non, 4)}/{round(np.average(psnr_non_list), 4)}  '
        #       f'ssim: {round(ssim_non, 4)}/{round(np.average(ssim_non_list), 4)}  '
        #       f'lpips: {round(lpips_non, 4)}/{round(np.average(lpips_non_list), 4)}  '
        #       f'rmse: {round(rmse_non, 4)}/{round(np.average(rmse_non_list), 4)}  '

        #       f'{len(psnr_all_list)}')

    print('-----------------------------------------------------------------------------')
    print(f'All psnr: {round(np.average(psnr_all_list), 4)} ssim: {round(np.average(ssim_all_list), 4)} lpips: {round(np.average(lpips_all_list), 4)} rmse: {round(np.average(rmse_all_list), 4)}')
    # print(f'Shadow psnr: {round(np.average(psnr_shadow_list), 4)} ssim: {round(np.average(ssim_shadow_list), 4)} lpips: {round(np.average(lpips_shadow_list), 4)} rmse: {round(np.average(rmse_shadow_list), 4)}')
    # print(f'Non psnr: {round(np.average(psnr_non_list), 4)} ssim: {round(np.average(ssim_non_list), 4)} lpips: {round(np.average(lpips_non_list), 4)} rmse: {round(np.average(rmse_non_list), 4)}')

########## Set the paths for evaluation ##########
# gt_root: ground truth root path
# pre_root: prediction root path
# mask_root: mask root path
# input_root: input root path (not used in the evaluation, only when you want to know the metrics of the input images)

########## General Shadow Removal Evaluation ##########
##### Example paths for ISTD+ dataset #####
mask_root = '/home/zhxing/Datasets/ISTD+/test/test_B_GT_NoSDDNet' # test_B_GT_NoSDDNet indicates the ground truth shadow mask here, not the one generated by SDDNet
gt_root = '/home/zhxing/Datasets/ISTD+/test/test_C'
input_root = '/home/zhxing/Datasets/ISTD+/test/test_A'
pred_root = '/home/zhxing/Projects/ShadowSurvey/ShadowRemoval/Auto/ISTD+512'

##### Example paths for SRD dataset #####
# mask_root = '/home/zhxing/Datasets/SRD_inpaint4shadow_fix/test/test_B_GT_NoSDDNet' # test_B_GT_NoSDDNet indicates the ground truth shadow mask here, not the one generated by SDDNet
# gt_root = '/home/zhxing/Datasets/SRD_inpaint4shadow_fix/test/test_C'
# input_root = '/home/zhxing/Datasets/SRD_inpaint4shadow_fix/test/test_A'
# pred_root = '/home/zhxing/Projects/ShadowSurvey/ShadowRemoval/Auto/SRD512'


########## Document Shadow Removal Evaluation ##########
##### Example paths for RDD dataset #####
# mask_root = None # There is no mask ground truth for document shadow removal dataset
# gt_root = '/home/zhxing/Datasets/RDD_data/test/gt'
# pred_root = '/home/zhxing/Projects/ShadowSurvey/DocShadowRemoval/BEDSR-Net'

# Start evaluation
evaluation(gt_root, pred_root, mask_root)
