import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from medpy import metric
from torchvision import transforms
import torch
import cv2

# Path to the directories containing the ground truth and predicted images
root_path = './VideoShadowDetection/STICT' # Prediction root dir
flow_root_path = '/home/zhxing/Datasets/ViSha/test/labels_flow_numpy/' # Optical flow path
gt_path = '/home/zhxing/Datasets/ViSha/test/labels' # Ground truth path

# Image file extensions
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_list(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                subname = path.split('/')
                images.append(os.path.join(subname[-2], subname[-1]))
    return images

def flow_warp(labels, flow):
    labels = labels.squeeze().numpy()
    h, w = labels.shape
    flow = flow.squeeze().permute(1, 2, 0).numpy()
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    flow_map[..., 0] = np.arange(w)
    flow_map[..., 1] = np.arange(h)[:, np.newaxis]
    flow_map += flow
    warped = cv2.remap(labels, flow_map, None, cv2.INTER_LINEAR)
    return torch.from_numpy(warped).unsqueeze(0)

def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure

def cal_temporal2(pred, gt, thr=1.0, g=0.0):
    assert pred.shape == gt.shape
    return np.sum(np.abs(pred - gt)) / (gt.size)

class AvgMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def computeBER_mth(gt_path, pred_path):
    print(gt_path, pred_path)
    gt_list = get_image_list(gt_path)
    nim = len(gt_list)
    
    stats_jaccard = np.zeros(nim, dtype='float')

    total_Tp = 0
    total_Tn = 0
    total_P = 0
    total_N = 0

    for i in tqdm(range(0, len(gt_list)), desc="Calculating Metrics:"):
        im = gt_list[i]
        GTim = np.asarray(Image.open(os.path.join(gt_path, im)).convert('L'))
        posPoints = GTim > 0.5
        sz = GTim.shape
        
        Predim = np.asarray(Image.open(os.path.join(pred_path, im.replace(".jpg", ".png"))).convert('L').resize((sz[1], sz[0]), Image.NEAREST))
        
        pred_array = np.array(Predim) > 255 * 0.5
        gd_array = np.array(GTim) > 0.5

        total_P += np.sum(gd_array)
        total_N += np.sum(~gd_array)
        total_Tp += np.sum(gd_array & pred_array)
        total_Tn += np.sum(~gd_array & ~pred_array)
        
        # IoU
        prediction = Predim / 255.
        gt = GTim / 255.

        pred = (prediction > 0.5)
        gt = (gt > 0.5)
        stats_jaccard[i] = metric.binary.jc(pred, gt)    
    
    # Print BER
    if total_P > 0 and total_N > 0:
        ber = 0.5 * (2 - total_Tp / total_P - total_Tn / total_N) * 100
        ber_pos = (1 - total_Tp / total_P) * 100
        ber_neg = (1 - total_Tn / total_N) * 100

        print(f'BER: {ber:.2f}%, BER_pos: {ber_pos:.2f}%, BER_neg: {ber_neg:.2f}%')
    else:
        print("Error: Division by zero in BER calculation")
    
    # Print IoU
    jaccard_value = np.mean(stats_jaccard)
    print('IoU:', jaccard_value)
    
    return 0


img_transform = transforms.Compose([
    transforms.ToTensor(),
])


# start to evaluate
all_name = os.listdir(gt_path) # Names of all videos

recon1avg = AvgMeter()
recon2avg = AvgMeter()
meanavg = AvgMeter()
precision_record, recall_record = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
Jaccard_record = AvgMeter()
BER_record = AvgMeter()
shadow_BER_record = AvgMeter()
non_shadow_BER_record = AvgMeter()
temporal_record = AvgMeter()
jaccardmeter2 = AvgMeter()

# for Temporal stability (TS) evaluation
for name in tqdm(all_name):
    all_img = os.listdir(root_path + '/' + name)
    img_num = len(all_img)
    all_img.sort()
    
    for idex in range(0, img_num):
        img_name1 = all_img[idex]
        img_path1 = root_path + '/' + name + '/' + img_name1
        labels1 = Image.open(img_path1).convert('L')
        l1 = img_name1.split('.')[0]

        labels1 = labels1.resize((512, 512), Image.NEAREST)
        if idex < img_num - 1:
            img_name2 = all_img[idex + 1]
            img_path2 = root_path + '/' + name + '/' + img_name2
            labels2 = Image.open(img_path2).convert('L')
            labels2 = labels2.resize((512, 512), Image.NEAREST)
            labels2 = np.array(labels2).astype(np.uint8)
            l2 = img_name2.split('.')[0]

            flow_name = l1 + '_' + l2 + '.npy'
            flow_path = flow_root_path + '/' + name + '/' + flow_name
            if os.path.exists(flow_path):
                flow1_2 = np.load(flow_path)
            else:
                flow_name = l1 + '_' + "%08d" % (int(l1) + 1) + '.npy'
                flow_path = flow_root_path + '/' + name + '/' + flow_name
                flow1_2 = np.load(flow_path)
            
            labels2 = img_transform(labels2)
            flow1_2 = torch.from_numpy(flow1_2)
            flow1_2 = flow1_2.permute(2, 0, 1).contiguous().unsqueeze(0)
            recon_labels1 = flow_warp(labels2.unsqueeze(dim=0), flow1_2).squeeze()
            recon_labels1 = recon_labels1.numpy()
            labels1 = np.array(labels1)

            temporal_record.update(cal_temporal2(recon_labels1, labels1, thr=1.0, g=0.0))

# for BER and IoU evaluation
pred_path = root_path # Prediction root dir
computeBER_mth(gt_path, pred_path)

log = 'Temporal Stability (TS):{}'.format(temporal_record.avg)
print(log)
