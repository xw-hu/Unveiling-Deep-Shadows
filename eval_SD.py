import os
import numpy as np
from PIL import Image
import argparse

def calculate_ber(pred_path, gd_path):
    total_Tp = 0
    total_Tn = 0
    total_P = 0
    total_N = 0

    pred_files = sorted(os.listdir(pred_path))
    gd_files = sorted(os.listdir(gd_path))

    for pred_file, gd_file in zip(pred_files, gd_files):
        pred_img = Image.open(os.path.join(pred_path, pred_file)).convert('L')
        gd_img = Image.open(os.path.join(gd_path, gd_file)).convert('L')

        # No need to resize the images to ground truth size, as they are already resized during inference
        # If your images are not resized during inference, you may need to resize them here
        # pred_img = pred_img.resize(gd_img.size, Image.NEAREST)


        pred_array = np.array(pred_img) > 255 * 0.5
        gd_array = np.array(gd_img) > 0.5

        total_P += np.sum(gd_array)
        total_N += np.sum(~gd_array)
        total_Tp += np.sum(gd_array & pred_array)
        total_Tn += np.sum(~gd_array & ~pred_array)

    ber = 0.5 * (2 - total_Tp / total_P - total_Tn / total_N) * 100
    ber_pos = (1 - total_Tp / total_P) * 100
    ber_neg = (1 - total_Tn / total_N) * 100

    print(f'BER: {ber:.2f}%, BER_pos: {ber_pos:.2f}%, BER_neg: {ber_neg:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate BER, BER_pos, and BER_neg")
    parser.add_argument('--pred_path', type=str, default="/home/zhxing/Projects/ShadowSurvey/ShadowDetection/DSC/demo_CUHK_CUHK512", help="Path to predicted images")
    parser.add_argument('--gd_path', type=str, default="/home/zhxing/Datasets/CUHKshadow-SDD/test/train_B", help="Path to ground truth images")

    # parser.add_argument('--pred_path', type=str, default="/home/zhxing/Projects/ShadowSurvey/ShadowDetection/DSC/demo_SBU-Refine_SBU-Refine512", help="Path to predicted images")
    # parser.add_argument('--gd_path', type=str, default="/home/zhxing/Datasets/SBU-refined/SBUTestNew/ShadowMasks", help="Path to ground truth images")

    # parser.add_argument('--pred_path', type=str, default="/home/zhxing/Projects/ShadowSurvey/ShadowDetection/DSC/demo_SRD_SBU-Refine512", help="Path to predicted images")
    # parser.add_argument('--gd_path', type=str, default="/home/zhxing/Datasets/SRD_inpaint4shadow_fix/test/test_B_GT_NoSDDNet", help="Path to ground truth images") # test_B_GT_NoSDDNet indicates the ground truth shadow mask here, not the one generated by SDDNet


    args = parser.parse_args()
    print("Pred: ", args.pred_path)
    print("GT: ", args.gd_path)
    calculate_ber(args.pred_path, args.gd_path)
