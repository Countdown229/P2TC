import argparse
import sys
sys.path.append("..")
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from models.ConResNet_v16_T41 import ConResNet
from dataset.HeartDataSet_730 import HeartValDataset
import os
from math import ceil
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2' 
def get_arguments():
    parser = argparse.ArgumentParser(description="ConResNet for 3D medical image segmentation.")
    parser.add_argument("--data-dir", type=str, default='/home/leij/disk1/pp_data/Pro1+/ConResNet-main',
                        help="Path to the directory containing your dataset.")
    parser.add_argument("--data-list", type=str, default='/home/leij/disk1/pp_data/Pro1+/ConResNet-main/dataset_ct.json',  #
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default='80,160,160',
                        help="Comma-separated string with depth, height and width of sub-volumnes.")
    parser.add_argument("--num-classes", type=int, default=8,
                        help="Number of classes to predict (ET, WT, TC).")
    parser.add_argument("--restore-from", type=str, default='/home/leij/disk1/pp_data/Pro1+/ConResNet-main/snapshots/conresnet/heartct/2024-12-24 20:51:55_T41_k5_L2Cbaseline/ConResNet_40000_0.8829339543978374.pth', #
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='1', #'0'
                        help="choose gpu device.")
    parser.add_argument("--weight-std", type=bool, default=True,
                        help="whether to use weight standarization in CONV layers.")
    return parser.parse_args()


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    deps_missing = target_size[0] - img.shape[2]
    rows_missing = target_size[1] - img.shape[3]
    cols_missing = target_size[2] - img.shape[4]
    padded_img = np.pad(img, ((0, 0), (0, 0),(0, deps_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding(net, img_list, tile_size, classes):
    image, image_res = img_list
    interp = nn.Upsample(size=tile_size, mode='trilinear', align_corners=True)
    image_size = image.shape
    overlap = 1/3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)  # strided convolution formula
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = torch.zeros((classes, image_size[2], image_size[3], image_size[4]), dtype=torch.float16)
    count_predictions = torch.zeros((classes, image_size[2], image_size[3], image_size[4]), dtype=torch.float16)

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                y1 = int(row * strideHW)
                x1 = int(col * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                y2 = min(y1 + tile_size[1], image_size[3])
                x2 = min(x1 + tile_size[2], image_size[4])
                d1 = max(int(d2 - tile_size[0]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]
                padded_img = pad_image(img, tile_size)
                padded_img_res = pad_image(img_res, tile_size)
                padded_prediction = net([torch.from_numpy(padded_img).cuda(), torch.from_numpy(padded_img_res).cuda()])
                padded_prediction = F.sigmoid(padded_prediction[0])

                padded_prediction = interp(padded_prediction).cpu().data[0]
                prediction = padded_prediction[0:img.shape[2],0:img.shape[3], 0:img.shape[4], :]
                count_predictions[:, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    #del count_predictions
    full_probs = full_probs.numpy().transpose(1,2,3,0)
    return full_probs

def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    num = np.sum(np.multiply(predict, target), axis=1)
    den = np.sum(predict, axis=1) + np.sum(target, axis=1) +1

    dice = 2*num / den
    return dice.mean()

def compute_per_class_mask_iou(gt_masks, pred_masks):
    """Computes per_class_IoU overlaps between two sets of masks.
    gt_masks, pred_masks: [Height, Width, Depth, instances], zero-padding if there's no such instance.
    Returns ious per instance.
    """
    # flatten masks and compute their areas
    gt_masks = np.reshape(gt_masks > .5, (-1, gt_masks.shape[-1])).astype(np.float32)
    gt_masks = torch.from_numpy(gt_masks).cuda()#
    pred_masks = np.reshape(pred_masks > .5, (-1, pred_masks.shape[-1])).astype(np.float32)
    pred_masks = torch.from_numpy(pred_masks).cuda()#
    area1 = torch.sum(gt_masks, dim=0)#
    area2 = torch.sum(pred_masks, dim=0)#
    intersections = torch.tensor([torch.mm(gt_masks.T, pred_masks)[i, i] for i in range(gt_masks.shape[-1])]).cuda()
    union = area1 + area2 - intersections
    ious = intersections /(union +1e-6)
    #del gt_masks, pred_masks, intersections, union
    ious = ious.cpu().numpy()#
    return ious

def compute_per_class_mask_dice(gt_masks, pred_masks):
    '''Computes per_class_dice overlaps between two sets of masks.
    gt_masks, pred_masks: [Height, Width, Depth, instances], zero-padding if there's no such instance.
    Returns ious per instance.
    '''
    # flatten masks and compute their areas
    gt_masks = np.reshape(gt_masks > .5, (-1, gt_masks.shape[-1])).astype(np.float32)
    gt_masks = torch.from_numpy(gt_masks).cuda()#
    pred_masks = np.reshape(pred_masks > .5, (-1, pred_masks.shape[-1])).astype(np.float32)
    pred_masks = torch.from_numpy(pred_masks).cuda()#
    area1 = torch.sum(gt_masks, dim=0)#
    area2 = torch.sum(pred_masks, dim=0)#
    intersections = torch.tensor([torch.mm(gt_masks.T, pred_masks)[i, i] for i in range(gt_masks.shape[-1])]).cuda()
    dice = 2 * intersections / (area1 + area2 + 1e-6)
    #del gt_masks, pred_masks, area1, area2, intersections
    dice = dice.cpu().numpy()
    return dice

def compute_mask_iou(gt_masks, pred_masks):
    """Computes IoU overlaps between two sets of masks. Regard different classes as the same.
    gt_masks, pred_masks: [Height, Width, Depth].
    Returns ious of the two masks.
    """
    # flatten masks and compute their areas
    gt_masks[gt_masks > 0] = 1
    pred_masks[pred_masks > 0] = 1
    gt_masks = np.reshape(gt_masks, (-1)).astype(np.int32)
    pred_masks = np.reshape(pred_masks, (-1)).astype(np.int32)
    area1 = np.sum(gt_masks)
    area2 = np.sum(pred_masks)
    # intersections and union
    intersections = np.dot(gt_masks.T, pred_masks)
    union = area1 + area2 - intersections
    ious = intersections / (union + 1e-6)  # avoid intersections to be divided by 0
    return ious

def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    d, h, w = map(int, args.input_size.split(','))

    input_size = (d, h, w)

    model = ConResNet(input_size, num_classes=args.num_classes, weight_std=args.weight_std)
    model = nn.DataParallel(model)

    print('loading from checkpoint: {}'.format(args.restore_from))
    if os.path.exists(args.restore_from):
        model.load_state_dict(torch.load(args.restore_from, map_location=torch.device('cpu')))
    else:
        print('File not exists in the reload path: {}'.format(args.restore_from))

    model.eval()
    model.cuda()

    testloader = data.DataLoader(
        HeartValDataset(args.data_dir, args.data_list, subset="test"),
        batch_size=1, shuffle=False, pin_memory=True)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    dices = []
    ious = []
    mask_iou = []
    total_time=0
    for index, batch in enumerate(testloader):
        image, image_res, label, size, name, affine = batch
        size = size[0].numpy()
        print("size:",size)
        affine = affine[0].numpy()
        print("name[0]:",str(name[0]))
        start_time = time.time()
        with torch.no_grad():
            output = predict_sliding(model, [image.numpy(),image_res.numpy()], input_size, args.num_classes)
        detect_time = time.time() - start_time
        seg_pred_8class = np.asarray(np.around(output), dtype=np.uint8) #[363,512,512,8]
        seg_pred = np.zeros_like(seg_pred_8class[:,:,:,0])
        for i in range(args.num_classes):
            seg_pred = np.where(seg_pred_8class[:, :, :, i] ==1, i, seg_pred)
        seg_gt = np.asarray(label[0].numpy()[:, :size[0], :size[1], :size[2]], dtype=np.int16)
        #del image, image_res, label, size, output
      
        seg_pred_8class = seg_pred_8class.transpose((1,2,0,3))   #(160, 512, 512, 8) -- (512,512,160,8)
        seg_gt = seg_gt.transpose((2,3,1,0))            #(8, 160, 512, 512) -- (512,512,160,8)
        print("transfered seg_gt: ",seg_gt.shape)
        print("transfered seg_pred_8class:",seg_pred_8class.shape)
        per_class_iou = compute_per_class_mask_iou(seg_gt[:,:,:,1:], seg_pred_8class[:,:,:,1:])
        ious.append(per_class_iou)
        per_class_dice = compute_per_class_mask_dice(seg_gt[:,:,:,1:], seg_pred_8class[:,:,:,1:])
        dices.append(per_class_dice)
        print("Processing ", name[0][-26:], ": Dice_i_mean = ", str(per_class_dice.mean()), ", Dice of MYO1,LA2,LV3,RA4,RV5,AA6,PA7:", per_class_dice)
        print("Each Seg time: ",detect_time)
        seg_pred = seg_pred.transpose((1,2,0))
        seg_pred = seg_pred.astype(np.int16)
        seg_pred = nib.Nifti1Image(seg_pred, affine=affine)
        seg_save_p = os.path.join("outputs/" +"ct_train_"+name[0][-17:-13] + "_label.nii.gz")
        # seg_save_p = os.path.join("outputs/" +"mr_train_"+name[0][-17:-13] + "_label.nii.gz")
        nib.save(seg_pred, seg_save_p)
        total_time += detect_time

        # wholeMaskIou = compute_mask_iou(label.detach().cpu().numpy(), seg_pred)
        # print("Mask iou: ", wholeMaskIou)
        # mask_iou.append(wholeMaskIou)
        # del seg_pred_8class, seg_pred, seg_gt
  
    dices = np.array(dices)
    dices_mean = round(dices.mean(),4)
    ious = np.array(ious)
    ious_mean = round(ious.mean(), 4)

    # mask_iou = np.array(mask_iou)
    # print("Mask Iou: ", mask_iou.mean())

    print("Average acore: Dice_mean: ", str(dices_mean), ", Dice of MYO1,LA2,LV3,RA4,RV5,AA6,PA7: ", np.mean(dices, axis=0))
    print("Average acore: Iou_mean: ", str(ious_mean), ", Iou of MYO1,LA2,LV3,RA4,RV5,AA6,PA7: ", np.mean(ious, axis=0))
    print("iou std:", np.std(ious, axis=0))
    print("dice std:", np.std(dices, axis=0))
    print("Total ious std mean:", np.std(ious, axis=0).mean())
    print("Total dice std mean:", np.std(dices, axis=0).mean())
    print("Total seg time: ", total_time)
if __name__ == '__main__':
    main()
    #nohup python -u test.py >test0.log 2>&1 &