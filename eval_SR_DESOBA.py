import os
import numpy as np
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from imageio.v2 import imread
import skimage
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import rgb2lab
import matplotlib.pyplot as plt


def load_item(gt_path, pre_path, mask_path, ignore_mask_path):

    gt = imread(gt_path)
    pre = imread(pre_path)
    mask = imread(mask_path)
    ignore_mask = imread(ignore_mask_path)

    # resize to gt size
    pre = resize(pre, (gt.shape[0], gt.shape[1]))
    mask = resize(mask, (gt.shape[0], gt.shape[1]))
    ignore_mask = resize(ignore_mask, (gt.shape[0], gt.shape[1]))

    mask = (mask > 255 * 0.9).astype(np.uint8) * 255
    ignore_mask = (ignore_mask > 255 * 0.9).astype(np.uint8) * 255

    return to_tensor(gt), to_tensor(pre), to_tensor(mask), to_tensor(ignore_mask)


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    img_t = img_t.unsqueeze(dim=0)
    return img_t


def resize(img, target_size):
    img = skimage.transform.resize(img, target_size, mode='reflect', anti_aliasing=True)
    img = (img * 255).astype(np.uint8)

    return img

def calc_rmse(real_img, fake_img, ignore_mask=None):
    # Convert images to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    
    if ignore_mask is not None:
        ignore_mask = ignore_mask.permute(0, 2, 3, 1)
        ignore_mask = ignore_mask.detach().cpu().numpy().astype(np.uint8)[0]

        # If the mask has a single channel, broadcast it across all channels
        if ignore_mask.shape[-1] == 1:
            ignore_mask = np.repeat(ignore_mask, 3, axis=-1)
        
        # Invert the mask to keep regions not ignored
        ignore_mask = ignore_mask == 0

        # Apply the mask to filter out the "don't care" regions
        real_lab = real_lab[ignore_mask]
        fake_lab = fake_lab[ignore_mask]

    if real_lab.size == 0 or fake_lab.size == 0:
        # Handle the case where no valid regions are left after masking
        return 0.0  # Return zero rmse if no valid regions are left

    # Calculate rmse
    rmse = np.sqrt(((real_lab - fake_lab) ** 2).mean())

    return rmse

def zhxing_psnr(gt, pre, ignore_mask):

    if ignore_mask is not None:
        ignore_mask = ignore_mask.permute(0, 2, 3, 1)
        ignore_mask = ignore_mask.detach().cpu().numpy().astype(np.uint8)[0]

        # If the mask has a single channel, broadcast it across all channels
        if ignore_mask.shape[-1] == 1:
            ignore_mask = np.repeat(ignore_mask, 3, axis=-1)
        
        # Invert the mask to keep regions not ignored
        ignore_mask = ignore_mask == 0

        # Apply the mask to filter out the "don't care" regions
        gt = gt[ignore_mask]
        pre = pre[ignore_mask]

    if gt.size == 0 or pre.size == 0:
        # Handle the case where no valid regions are left after masking
        return float('inf'), 1.0, 0.0  # Return max PSNR, perfect SSIM, and zero rmse

    # Calculate PSNR, SSIM, and rmse only for the regions of interest
    psnr = compare_psnr(gt, pre, data_range=255)

    return psnr


def metric(gt, pre, ignore_mask=None):
    pre = pre * 255.0
    pre = pre.permute(0, 2, 3, 1)
    pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

    gt = gt * 255.0
    gt = gt.permute(0, 2, 3, 1)
    gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

    psnr = zhxing_psnr(gt, pre, ignore_mask)
    rmse = calc_rmse(gt, pre, ignore_mask)

    return psnr, 0, rmse

def visualize_mask_application(gt, pre, ignore_mask):
    gt = gt.squeeze().permute(1, 2, 0).cpu().numpy()
    pre = pre.squeeze().permute(1, 2, 0).cpu().numpy()

    if ignore_mask is not None:
        ignore_mask = ignore_mask.squeeze().cpu().numpy()
        
        # Expand the ignore_mask to have the same number of channels as gt
        ignore_mask = np.expand_dims(ignore_mask, axis=-1)
        ignore_mask = np.repeat(ignore_mask, 3, axis=-1)

        # Create the inverted mask
        invert_ignore_mask = ignore_mask == 0
        
        gt_masked = gt * ignore_mask
        pre_masked = pre * ignore_mask
        
        # Apply the inverted mask
        gt_inverted_masked = gt * invert_ignore_mask
        pre_inverted_masked = pre * invert_ignore_mask
    else:
        gt_masked = gt
        pre_masked = pre
        gt_inverted_masked = gt
        pre_inverted_masked = pre

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(gt)
    axes[0].set_title("Original GT")
    axes[1].imshow(pre)
    axes[1].set_title("Original Prediction")
    axes[2].imshow(gt_masked)
    axes[2].set_title("Masked GT")
    axes[3].imshow(gt_inverted_masked)
    axes[3].set_title("Inverted Masked GT")
    axes[4].imshow(pre_inverted_masked)
    axes[4].set_title("Inverted Masked Prediction")
    plt.show()


def evaluation(gt_root, pre_root, mask_root, demask_root, vis_flag):
    fnames = os.listdir(gt_root)
    fnames.sort()

    psnr_all_list, ssim_all_list, rmse_all_list = [], [], []
    psnr_non_list, ssim_non_list, rmse_non_list = [], [], []
    psnr_shadow_list, ssim_shadow_list, rmse_shadow_list = [], [], []

    for fname in fnames:
        # print(fname)
        gt_path = os.path.join(gt_root, fname)
        pre_path = os.path.join(pre_root, fname)
        mask_path = os.path.join(mask_root, fname)
        ignore_mask_path = os.path.join(demask_root, fname)

        mask_path = mask_path.replace('.jpg', '.png')
        # pre_path = pre_path.replace('.jpg', '.png')
        ignore_mask_path = ignore_mask_path.replace('.jpg', '.png')

        if not os.path.exists(mask_path):
            print(f'Mask path {mask_path} does not exist. Skipping...')
            continue

        gt, pre, mask, ignore_mask = load_item(gt_path, pre_path, mask_path, ignore_mask_path)

        if vis_flag:
            visualize_mask_application(gt, pre, ignore_mask)


        psnr_all, ssim_all, rmse_all = metric(gt, pre, ignore_mask=ignore_mask)

        psnr_all_list.append(psnr_all)
        ssim_all_list.append(ssim_all)
        rmse_all_list.append(rmse_all)


    print(f'All psnr: {round(np.average(psnr_all_list), 4)} rmse: {round(np.average(rmse_all_list), 4)}')
    print('-----------------------------------------------------------------------------')


##### evaluation start, replace the following paths with your own paths #####
mask_root = '/home/zhxing/Datasets/DESOBA_xvision/test/test_B_GT_NoSDDNet' # test_B_GT_NoSDDNet indicates the ground truth shadow mask here, not the one generated by SDDNet
gt_root = '/home/zhxing/Datasets/DESOBA_xvision/test/test_C'
input_root = '/home/zhxing/Datasets/DESOBA_xvision/test/test_A'
demask_root = '/home/zhxing/Datasets/DESOBA_xvision/InstanceMask'
pred_root = '/home/zhxing/Projects/ShadowSurvey/ShadowRemoval/BMNet/SRD512_DESOBA'

evaluation(gt_root, pred_root, mask_root, demask_root, vis_flag=False)
