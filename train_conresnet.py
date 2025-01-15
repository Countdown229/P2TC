import argparse
import sys
sys.path.append("..")
import time
import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from apex import amp
import os
import os.path as osp
from models.ConResNet_v16_T41 import ConResNet
from dataset.HeartDataSet_730 import HeartDataset, HeartValDataset
import timeit
from tensorboardX import SummaryWriter
from utils import loss
from utils.engine import Engine
from math import ceil
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
start = timeit.default_timer()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ConResNet for 3D Medical Image Segmentation.")

    parser.add_argument("--data_dir", type=str, default='/home/sunze/pp_data/ConResNet-main/')
    parser.add_argument("--train_list", type=str, default='/home/sunze/pp_data/ConResNet-main/dataset.json')
    parser.add_argument("--val_list", type=str, default='/home/sunze/pp_data/ConResNet-main/dataset.json')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/conresnet/heartct/')
    parser.add_argument("--reload_path", type=str, default='snapshots/conresnet/ConResNet_40000.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--input_size", type=str, default='80,160,160')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2*num / den

    return dice.mean()


# def compute_dice_score(preds, labels):

#     preds = F.sigmoid(preds)

#     pred_ET = preds[:, 0, :, :, :]
#     pred_WT = preds[:, 1, :, :, :]
#     pred_TC = preds[:, 2, :, :, :]
#     label_ET = labels[:, 0, :, :, :]
#     label_WT = labels[:, 1, :, :, :]
#     label_TC = labels[:, 2, :, :, :]
#     dice_ET = dice_score(pred_ET, label_ET).cpu().data.numpy()
#     dice_WT = dice_score(pred_WT, label_WT).cpu().data.numpy()
#     dice_TC = dice_score(pred_TC, label_TC).cpu().data.numpy()
#     return dice_ET, dice_WT, dice_TC
def compute_dice_score(preds, labels):
    # preds = F.sigmoid(preds)
    preds = torch.sigmoid(preds)
    # pred_BG = preds[:, 0, :,:,:]
    pred_MYO1=preds[:, 1, :,:,:]
    pred_LA2 =preds[:, 2, :,:,:]
    pred_LV3 =preds[:, 3, :,:,:]
    pred_RA4 =preds[:, 4, :,:,:]
    pred_RV5 =preds[:, 5, :,:,:]
    pred_AA6 =preds[:, 6, :,:,:]
    pred_PA7 =preds[:, 7, :,:,:]
    # # for test (don't need)
    # label_MYO1=labels[:, 1, :,:,:]
    # label_LA2 =labels[:, 2, :,:,:]
    # label_LV3 =labels[:, 3, :,:,:]
    # label_RA4 =labels[:, 4, :,:,:]
    # label_RV5 =labels[:, 5, :,:,:]
    # label_AA6 =labels[:, 6, :,:,:]
    # label_PA7 =labels[:, 7, :,:,:]

    # for train (can save 1000MB memory)
    label_MYO1=(labels==1)
    label_LA2 =(labels==2)
    label_LV3 =(labels==3)
    label_RA4 =(labels==4)
    label_RV5 =(labels==5)
    label_AA6 =(labels==6)
    label_PA7 =(labels==7)

    # dice_BG = dice_score(pred_BG, label_BG).cpu().data.numpy()
    dice_MYO1=dice_score(pred_MYO1, label_MYO1).cpu().data.numpy()
    dice_LA2 =dice_score(pred_LA2, label_LA2).cpu().data.numpy()
    dice_LV3 =dice_score(pred_LV3, label_LV3).cpu().data.numpy()
    dice_RA4 =dice_score(pred_RA4, label_RA4).cpu().data.numpy()
    dice_RV5 =dice_score(pred_RV5, label_RV5).cpu().data.numpy()
    dice_AA6 =dice_score(pred_AA6, label_AA6).cpu().data.numpy()
    dice_PA7 =dice_score(pred_PA7, label_PA7).cpu().data.numpy()
    del pred_MYO1, pred_LA2, pred_LV3, pred_RA4, pred_RV5, pred_AA6, pred_PA7, label_MYO1, label_LA2, label_LV3, label_RA4, label_RV5,label_AA6,label_PA7
    # return dice_BG, dice_MYO1, dice_LA2, dice_LV3, dice_RA4, dice_RV5, dice_AA6, dice_PA7
    return dice_MYO1, dice_LA2, dice_LV3, dice_RA4, dice_RV5, dice_AA6, dice_PA7

def predict_sliding(net, imagelist, tile_size, classes):
    image, image_res = imagelist
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    full_probs = torch.from_numpy(full_probs).cuda()
    count_predictions = torch.from_numpy(count_predictions).cuda()

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]
                prediction = net([img, img_res])
                prediction = prediction[0]

                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    del count_predictions
    return full_probs

def validate(input_size, model, ValLoader, num_classes):
    # val_BG = 0.0
    val_MYO1 = 0.0
    val_LA2 = 0.0
    val_LV3 = 0.0 
    val_RA4 = 0.0 
    val_RV5 = 0.0
    val_AA6 = 0.0 
    val_PA7 = 0.0
    for index, batch in enumerate(ValLoader):
        print('%d processd'%(index))
        image, image_res, label, size, name, affine = batch
        image = image.cuda()
        image_res = image_res.cuda()
        
        
        with torch.no_grad():
            pred = predict_sliding(model, [image, image_res], input_size, num_classes)
            del image, image_res
            label = label.cuda()
            # dice_BG, dice_MYO1, dice_LA2, dice_LV3, dice_RA4, dice_RV5, dice_AA6, dice_PA7 = compute_dice_score(pred, label)
            dice_MYO1, dice_LA2, dice_LV3, dice_RA4, dice_RV5, dice_AA6, dice_PA7 = compute_dice_score(pred, label) #[1, 8, 297, 512, 512]  [1, 8, 297, 512, 512]
            del label, pred
            val_MYO1+= dice_MYO1
            val_LA2 += dice_LA2
            val_LV3 += dice_LV3
            val_RA4 += dice_RA4
            val_RV5 += dice_RV5
            val_AA6 += dice_AA6
            val_PA7 += dice_PA7
    return val_MYO1/(index+1), val_LA2/(index+1),val_LV3/(index+1), val_RA4/(index+1), val_RV5/(index+1), val_AA6/(index+1), val_PA7/(index+1)
    # return val_BG/(index+1), val_MYO1/(index+1), val_LA2/(index+1),val_LV3/(index+1), val_RA4/(index+1), val_RV5/(index+1), val_AA6/(index+1), val_PA7/(index+1)

def main():
    """Create the ConResNet model and then start the training."""
    parser = get_arguments()
    print(parser)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)
        start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        writer = SummaryWriter(args.snapshot_dir+ str(start_datetime))

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)  #(80, 160, 160)

        cudnn.benchmark = True
        seed = args.random_seed #1234
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model = ConResNet(input_size, num_classes=args.num_classes, weight_std=True)
        model.train()
        device = torch.device('cuda:{}'.format(args.local_rank))
        
        model.to(device)  # 

        optimizer = optim.AdamW(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate}],
            lr=args.learning_rate, weight_decay=args.weight_decay)

        if args.num_gpus > 1:
            model = engine.data_parallel(model)        

        

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        loss_D = loss.DiceLoss4BraTS().to(device)
        loss_BCE = loss.BCELoss4BraTS().to(device)
        loss_B = loss.BCELossBoud().to(device)

        
        # if not os.path.exists("snapshots/conresnet/heartct/" + str(start_datetime)):
        #     os.makedirs("snapshots/conresnet/heartct/" + str(start_datetime))
        # if not os.path.exists(args.snapshot_dir):
        #     os.makedirs(args.snapshot_dir)
        
        trainloader, train_sampler = engine.get_train_loader(HeartDataset(args.data_dir, args.train_list, max_iters=args.num_steps * args.batch_size, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, subset="train"))
        valloader, val_sampler = engine.get_test_loader(HeartValDataset(args.data_dir, args.val_list, subset="val"))


        for i_iter, batch in enumerate(trainloader):
            i_iter += args.start_iters
            images, images_res, labels, labels_res = batch  #[2,1,80,160,160] [2,8,80,160,160]
            images = images.cuda()
            images_res = images_res.cuda()
            

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, i_iter, args.learning_rate, args.num_steps, args.power)

            preds= model([images, images_res])
            preds_seg = preds[0]    #[2,8,80,160,160]   
            preds_res = preds[1]    #[2,8,80,160,160]
            preds_resx2 = preds[2]  #[2,8,80,160,160]
            preds_resx4 = preds[3]  #[2,8,80,160,160]

            del images, images_res
            labels = labels.cuda()
            labels_res = labels_res.cuda()

            term_seg_Dice = loss_D.forward(preds_seg, labels)   #[2,8,80,160,160]   0.9141
            term_seg_BCE = loss_BCE.forward(preds_seg, labels)  #[2,8,80,160,160]   26.4128

            term_res_BCE = loss_B.forward(preds_res, labels_res)    #[2,8,80,160,160]   2.7299
            term_resx2_BCE = loss_B.forward(preds_resx2, labels_res)#[2,8,80,160,160]   2.6555
            term_resx4_BCE = loss_B.forward(preds_resx4, labels_res)#[2,8,80,160,160]   2.3624
            del preds, preds_seg, preds_res, preds_resx2, preds_resx4, labels, labels_res

            w0, w1, w2, w3 = 2, 1, 0.5, 0.5  ###Change Weight
            term_all = w0 * (term_seg_Dice+term_seg_BCE) + w1 * term_res_BCE + w2 * term_resx2_BCE + w3 * term_resx4_BCE
            # Original
            # term_all = (term_seg_Dice + term_seg_BCE) + term_res_BCE + 0.5 * (term_resx2_BCE +term_resx4_BCE)
            term_all.backward()

            optimizer.step()

            if i_iter % 200 == 0 and (args.local_rank == 0):
                writer.add_scalar('learning_rate', lr, i_iter)
                writer.add_scalar('loss', term_all.cpu().data.numpy(), i_iter)
            if i_iter % 10 == 0:
                print('iter = {} of {} completed, lr = {:.4}, seg_loss = {:.4}, res_loss = {:.4}'.format(
                    i_iter, args.num_steps, lr, (term_seg_Dice+term_seg_BCE).cpu().data.numpy(), (term_res_BCE+term_resx2_BCE+term_resx4_BCE).cpu().data.numpy()))

            # val
            if i_iter % args.val_pred_every == 0:# and i_iter != 0:
                print('validate ...')
                # val_BG,val_MYO1,val_LA2,val_LV3,val_RA4,val_RV5,val_AA6,val_PA7 = validate(input_size, model, valloader, args.num_classes)
                val_MYO1,val_LA2,val_LV3,val_RA4,val_RV5,val_AA6,val_PA7 = validate(input_size, model, valloader, args.num_classes)
                val_mean = (val_MYO1+val_LA2+val_LV3+val_RA4+val_RV5+val_AA6+val_PA7)/7
                if (args.local_rank == 0):
                    # writer.add_scalar('Val_BG_Dice', val_BG, i_iter)
                    writer.add_scalar('Val_MYO1_Dice', val_MYO1, i_iter)
                    writer.add_scalar('Val_LA2_Dice', val_LA2, i_iter)
                    writer.add_scalar('Val_LV3_Dice', val_LV3, i_iter)
                    writer.add_scalar('Val_RA4_Dice', val_RA4, i_iter)
                    writer.add_scalar('Val_RV5_Dice', val_RV5, i_iter)
                    writer.add_scalar('Val_AA6_Dice', val_AA6, i_iter)
                    writer.add_scalar('Val_PA7_Dice', val_PA7, i_iter)
                    # print('Validate iter = {}, BG = {:.2}, MYO1 = {:.2}, LA2 = {:.2}, LV3 = {:.2}, RA4 = {:.2}, RV5 = {:.2}, AA6 = {:.2}, PA7 = {:.2}'.format(i_iter, val_BG ,val_MYO1,val_LA2,val_LV3,val_RA4,val_RV5,val_AA6,val_PA7))
                    print('Validate iter = {}, MYO1 = {:.2}, LA2 = {:.2}, LV3 = {:.2}, RA4 = {:.2}, RV5 = {:.2}, AA6 = {:.2}, PA7 = {:.2}'.format(i_iter, val_MYO1,val_LA2,val_LV3,val_RA4,val_RV5,val_AA6,val_PA7))

            if i_iter >= args.num_steps - 1 and (args.local_rank == 0):
                print('save last model ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, str(start_datetime), 'ConResNet_' + str(args.num_steps) + "_" + str(val_mean) + '.pth'))
                break

            if i_iter % args.val_pred_every == 0 and i_iter!=0 and (args.local_rank == 0):
                print('save model ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, str(start_datetime), 'ConResNet_' + str(i_iter) + "_" + str(val_mean) + '.pth'))

            
    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()