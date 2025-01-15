import os
import json
import time
import numpy as np
import nibabel as nib
import random
from torch.utils import data
from skimage.transform import resize
import torch
class HeartDataset(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(80,160,160), scale=True, mirror=True, subset="train",ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.is_mirror = mirror
        self.subset = subset
        self.use_res = True
        # Train or validation dataset?
        assert subset in ["train", "val"]

        # Load dataset info
        info = json.load(open(self.list_path))

        if subset == "train":
            #info = info[13:]
            info = list(info['train'])
        else:
            #info = info[:13]
            info = list(info['val'])
        self.img_ids = [i_id for i_id in info]

        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        
        self.files=[]
        # Add images and masks
        for item in self.img_ids:
            self.files.append({
                "image_id":item['image_id'],
                "image": os.path.join(self.root, item['path']),
                "label": os.path.join(self.root, item["label"])
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((8, shape[0], shape[1], shape[2])).astype(np.float32)
        a = (label == 1)
        b = (label == 2)
        c = (label == 3)
        d = (label == 4)
        e = (label == 5)
        f = (label == 6)
        g = (label == 7)
        background = (label == 0)
        results_map[0,:,:,:] = np.where(background, 1, 0)
        results_map[1, :, :, :] = np.where(a, 1, 0)
        results_map[2, :, :, :] = np.where(b, 1, 0)
        results_map[3, :, :, :] = np.where(c, 1, 0)
        results_map[4, :, :, :] = np.where(d, 1, 0)
        results_map[5, :, :, :] = np.where(e, 1, 0)
        results_map[6, :, :, :] = np.where(f, 1, 0)
        results_map[7, :, :, :] = np.where(g, 1, 0)
        return results_map

    # def truncate(self, MRI):
    #     Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
    #     idexs = np.argwhere(Hist >= 50)
    #     idex_max = np.float32(idexs[-1, 0])
    #     MRI[np.where(MRI >= idex_max)] = idex_max
    #     sig = MRI[0, 0, 0]
    #     MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
    #     MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
    #     return MRI

    def __getitem__(self, index):
        datafiles = self.files[index]
        imageNII = nib.load(datafiles["image"])
        label = nib.load(datafiles["label"]).get_data().copy()
        
        #image = self.truncate(imageNII.get_data().copy())   #use or not
        image = np.array([imageNII.get_data().copy()])
        del imageNII
        label = np.array(label)
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        if self.scale:
            scaler = np.random.uniform(0.9, 1.1)
        else:
            scaler = 1
        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape

        d_off = random.randint(0, abs(img_d - scale_d))
        h_off = random.randint(0, abs(img_h - scale_h))
        w_off = random.randint(0, abs(img_w - scale_w))

        image = image[:, h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]
        label = label[h_off: h_off + scale_h, w_off: w_off + scale_w, d_off: d_off + scale_d]

        label = self.id2trainId(label)
        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        label = label.transpose((0, 3, 1, 2))     # Depth x H x W
        # image = image.transpose((2, 0, 1))  # Channel x Depth x H x W
        # label = label.transpose((2, 0, 1))     # Depth x H x W
        
        if self.is_mirror:
            randi = np.random.rand(1)
            if randi <= 0.3:
                pass
            elif randi <= 0.4:
                image = image[:, :, :, ::-1]
                label = label[:, :, :, ::-1]
            elif randi <= 0.5:
                image = image[:, :, ::-1, :]
                label = label[:, :, ::-1, :]
            elif randi <= 0.6:
                image = image[:, ::-1, :, :]
                label = label[:, ::-1, :, :]
            elif randi <= 0.7:
                image = image[:, :, ::-1, ::-1]
                label = label[:, :, ::-1, ::-1]
            elif randi <= 0.8:
                image = image[:, ::-1, :, ::-1]
                label = label[:, ::-1, :, ::-1]
            elif randi <= 0.9:
                image = image[:, ::-1, ::-1, :]
                label = label[:, ::-1, ::-1, :]
            else:
                image = image[:, ::-1, ::-1, ::-1]
                label = label[:, ::-1, ::-1, ::-1]

        if self.scale:
            image = resize(image, (1, self.crop_d, self.crop_h, self.crop_w), order=1, mode='constant', cval=0, clip=True, preserve_range=True)
            label = resize(label, (8, self.crop_d, self.crop_h, self.crop_w), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        # Version2: New, Ours
        # # image -> res
        # image_res = np.zeros((1, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        # # 计算当前切片与前一个切片的残差
        # image_res[:, 1:, :, :] += np.abs(image[:, 1:self.crop_d, :, :] - image[:, :-1, :, :])
        # # 计算当前切片与后一个切片的残差
        # image_res[:, :-1, :, :] += np.abs(image[:, :-1, :, :] - image[:, 1:self.crop_d, :, :])
        # # label -> res
        # label_res = np.zeros((8, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
        # # 计算当前切片与前一个切片的残差
        # label_res[:, 1:, :, :] += np.abs(label[:, 1:self.crop_d, :, :] - label[:, :-1, :, :])
        # # 计算当前切片与后一个切片的残差
        # label_res[:, :-1, :, :] += np.abs(label[:, :-1, :, :] - label[:, 1:self.crop_d, :, :])
        # label_res[np.where(label_res == 0)] = 0
        # label_res[np.where(label_res != 0)] = 1
        if self.use_res:
            # image -> res
            image_res = np.zeros((1, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
            f= np.zeros_like(image_res).astype(np.float32)
            b= np.zeros_like(image_res).astype(np.float32)
            f[:, 1:, :, :] = np.abs(image[:, 1:self.crop_d, :, :] - image[:, 0:-1, :, :])
            b[:, :-1, :,:] = np.abs(image[:, 0:self.crop_d-1, :, :] - image[:, 1:self.crop_d, :, :])
            mask = (f!=0) & (b!=0)
            image_res[mask] += (f[mask] + b[mask]) * 0.5
            # print("image: ",image.shape)
            # print("res: ", image_res.shape)
            # print("f non-zero: ",  np.count_nonzero(f[f!=0]))
            # print("b non-zero: ", np.count_nonzero(b[b!=0]))
            # print("mask count: ", np.count_nonzero(mask))

            # label -> res
            label_res = np.zeros((8, self.crop_d, self.crop_h, self.crop_w)).astype(np.float32)
            fl = np.zeros_like(label_res).astype(np.float32)
            bl = np.zeros_like(label_res).astype(np.float32)
            fl[:, 1:, :, :] = np.abs(label[:, 1:self.crop_d, :, :] - label[:, 0:self.crop_d-1, :, :])
            bl[:, :-1, :,:] = np.abs(label[:, 0:self.crop_d-1, :, :] - label[:, 1:self.crop_d, :, :])
            maskl = (fl!=0) & (bl!=0)
            label_res[maskl] = 1
            # print("image: ",image.shape)
            # print("res: ", image_res.shape)
            # print("f non-zero: ",  np.count_nonzero(fl[fl!=0]))
            # print("b non-zero: ", np.count_nonzero(bl[bl!=0]))
            # print("mask count: ", np.count_nonzero(maskl))
            # print("label_res non-zero: ", np.count_nonzero(label_res))
            # print("label non-zero: ", np.count_nonzero(label))

            return image.copy(), image_res.copy(), label.copy(), label_res.copy()
        else:
            return image.copy(), label.copy(), 


class HeartValDataset(data.Dataset):
    def __init__(self, root, list_path, subset):
        self.root = root
        self.list_path = list_path
        self.subset = subset
        self.files=[]
        self.use_res = True
        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        # Load dataset info
        info = json.load(open(self.list_path))
        if subset == "train":
            info = list(info["train"])  #datamr/test/   label:0-7
        elif subset == "val":
            info = list(info["val"])    #datamr/train/  label:maskey
        elif subset == "test":
            info = list(info["test"])
        else:
            print("Subset set wrong!!")
        self.img_ids = [i_id for i_id in info]
        print("ValDataset:",len(self.img_ids))
        
        # Add images and masks
        for item in self.img_ids:
            self.files.append({
                "image_id":item['image_id'],
                "image": os.path.join(self.root, item['path']),
                "label": os.path.join(self.root, item["label"])
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)
    
    def id2trainId(self, label):
        shape = label.shape
        results_map = np.zeros((8, shape[0], shape[1], shape[2])).astype(np.float32)
        a = (label == 1)
        b = (label == 2)
        c = (label == 3)
        d = (label == 4)
        e = (label == 5)
        f = (label == 6)
        g = (label == 7)
        background = (label == 0)
        results_map[0,:,:,:] = np.where(background, 1, 0)
        results_map[1, :, :, :] = np.where(a, 1, 0)
        results_map[2, :, :, :] = np.where(b, 1, 0)
        results_map[3, :, :, :] = np.where(c, 1, 0)
        results_map[4, :, :, :] = np.where(d, 1, 0)
        results_map[5, :, :, :] = np.where(e, 1, 0)
        results_map[6, :, :, :] = np.where(f, 1, 0)
        results_map[7, :, :, :] = np.where(g, 1, 0)
        return results_map

    def truncate(self, MRI):
        Hist, _ = np.histogram(MRI, bins=int(MRI.max()))
        idexs = np.argwhere(Hist >= 50)
        idex_max = np.float32(idexs[-1, 0])
        MRI[np.where(MRI >= idex_max)] = idex_max
        sig = MRI[0, 0, 0]
        MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
        MRI = np.where(MRI != sig, MRI / np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
        return MRI
    
    def __getitem__(self, index):
        datafiles = self.files[index]
        imageNII = nib.load(datafiles["image"])
        labelNII = nib.load(datafiles["label"])
        label = labelNII.get_data().copy()
        
        # image = self.truncate(imageNII.get_data().copy())   #use or not
        image = np.array([imageNII.get_data().copy()])
        name = datafiles["image"]   #image path
        affine = labelNII.affine
        del imageNII, labelNII

        # # # for test
        # label = self.id2trainId(label)
        # label = label.transpose((0, 3, 1, 2))     # Depth x H x W

        # # for train
        label = label.transpose((2, 0, 1))

        image = image.transpose((0, 3, 1, 2))  # Channel x Depth x H x W
        image = image.astype(np.float32)
        label = label.astype(np.float32)
        
        size = image.shape[1:]
        
        if self.use_res:
            # image -> res
            image_res = np.zeros_like(image).astype(np.float32)
            f= np.zeros_like(image).astype(np.float32)
            b= np.zeros_like(image).astype(np.float32)
            f[:, 1:, :, :] = np.abs(image[:, 1:, :, :] - image[:, :-1, :, :])
            b[:, :-1, :,:] = np.abs(image[:, :-1, :, :] - image[:, 1:, :, :])
            mask = (f!=0) & (b!=0)
            image_res[mask] += (f[mask] + b[mask]) *0.5

            return image.copy(), image_res.copy(), label.copy(), np.array(size), name, affine
        else:
            return image.copy(),label.copy(), np.array(size), name, affine


        #foreward + backward
        # Version1: false
        # image_res = np.abs(image[:, 1:, :, :] - image_copyf[:, :-1, :, :]) + np.abs(image[:, :-1, :, :] - image_copyb[:, 1:, :, :])
        
        # Version2: Now Ours
        # cha, dep, hei, wei = image.shape
        # image_copyf = np.zeros_like(image).astype(np.float32)
        # image_copyf[:, 1:, :, :] = image[:, 0: dep-1, :, :]
        # image_copyb = np.zeros_like(image).astype(np.float32)
        # image_copyb[:, :-1, :, :] = image[:, 1:dep, :, :]
        # image_res = np.zeros((cha,dep,hei,wei)).astype(np.float32)
        # image_res[:, 1:, :, :] += np.abs(image[:, 1:dep, :, :] - image_copyf[:, :-1, :, :])
        # image_res[:, :-1, :, :]+= np.abs(image[:, 0:dep-1, :, :] - image_copyb[:, 1:, :, :])
        # del image_copyf, image_copyb

        

        # print("image: ",image.shape)
        # print("res: ", image_res.shape)
        # print("f non-zero: ",  np.count_nonzero(f[f!=0]))
        # print("b non-zero: ", np.count_nonzero(b[b!=0]))
        # print("mask count: ", np.count_nonzero(mask))

        
    
if __name__=="__main__":
    from torch.utils.data import DataLoader
    root = "/home/sunze/pp_data/ConResNet-main"
    list_path = "/home/sunze/pp_data/ConResNet-main/dataset_ct.json"
    dataset = HeartValDataset(root, list_path, subset="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        # 获取输入图像和标签
        images,images_res, labels,_ = data
        break

        # 在这里执行你想要的操作，比如在训练模型时进行前向传播和损失计算

        # 打印当前批次的形状
        print("Batch {}, images shape: {}, labels shape: {}".format(i+1, images.shape, labels.shape))