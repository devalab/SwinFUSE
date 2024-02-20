import json
import os
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from monai.config import print_config
from monai.data import CacheDataset, Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import PatchEmbed, UnetrBasicBlock
from monai.networks.nets import UNETR, SwinUNETR
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandSpatialCropd,
    NormalizeIntensityd,
    RandScaleIntensityd,
)
from monai.utils.enums import MetricReduction
from tqdm import tqdm
from monai.networks.nets.swin_unetr import SwinUNETRCoAttn

DATA_DIR = "/scratch/MSD/"
MODEL_DIR = "/scratch/MSD/logs/pre_train/swin/runs/"
LOG_DIR = "/scratch/MSD/logs/fine_tune/"
JSON_DIR = LOG_DIR + "brats21_folds.json"

use_pretrained = True
pretrained_path = os.path.normpath(MODEL_DIR + "model_bestValRMSE.pt")
DEVICE_IDS = [0, 1]
CACHE_RATE = 0.2
N_W = 3


class AverageMeter(object):
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
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(LOG_DIR, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def get_loader(batch_size, data_dir, json_list, fold, roi):
    data_dir = data_dir
    datalist_json = json_list
    train_files, validation_files = datafold_read(
        datalist=datalist_json, basedir=data_dir, fold=fold
    )
    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[roi[0], roi[1], roi[2]],
            ),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[roi[0], roi[1], roi[2]],
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    train_ds = CacheDataset(data=train_files, transform=train_transform, cache_rate=0.3)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=N_W,
        pin_memory=False,
        drop_last=False,
    )
    val_ds = CacheDataset(
        data=validation_files, transform=val_transform, cache_rate=0.4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=N_W,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader


roi = (96, 96, 96)
batch_size = 4
sw_batch_size = 16
fold = 1
infer_overlap = 0.5
max_epochs = 100
val_every = 10
train_loader, val_loader = get_loader(batch_size, DATA_DIR, JSON_DIR, fold, roi)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETRCoAttn(
    img_size=roi,
    in_channels=1,
    out_channels=3,
    feature_size=48,
    drop_rate=0.2,
    attn_drop_rate=0.2,
    dropout_path_rate=0.0,
)


if use_pretrained is True:
    print("Loading Weights from the Path {}".format(pretrained_path))
    weights = torch.load(pretrained_path)
    old_sd = weights["state_dict"].keys()
    new_sd = {}
    for k in old_sd:
        new_sd[k.replace("module.", "")] = weights["state_dict"][k]

    model.load_state_dict(new_sd, strict=False)

    model.swinViT.patch_embed = PatchEmbed(
        patch_size=model.swinViT.patch_size,
        in_chans=4,
        embed_dim=48,
        norm_layer=nn.LayerNorm,
        spatial_dims=3,
    )
    model.encoder1 = UnetrBasicBlock(
        spatial_dims=3,
        in_channels=4,
        out_channels=48,
        kernel_size=3,
        stride=1,
        norm_name="instance",
        res_block=True,
    )
    del weights, new_sd, old_sd
    print("Pretrained Weights Succesfully Loaded !")

model = nn.DataParallel(model, device_ids=DEVICE_IDS)
model.to(device)

torch.backends.cudnn.benchmark = True
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)
dice_acc = DiceMetric(
    include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True
)
model_inferer = partial(
    sliding_window_inference,
    roi_size=[roi[0], roi[1], roi[2]],
    sw_batch_size=sw_batch_size,
    predictor=model,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


def train_epoch(model, loader, optimizer, epoch, loss_func):
    model.train()
    # start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(device), batch_data["label"].to(device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=batch_size)
        # print(
        #     "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
        #     "loss: {:.4f}".format(run_loss.avg),
        #     "time {:.2f}s".format(time.time() - start_time),
        # )
        # start_time = time.time()
    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
):
    model.eval()
    # start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(device), batch_data["label"].to(
                device
            )
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(post_sigmoid(val_pred_tensor))
                for val_pred_tensor in val_outputs_list
            ]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            # dice_tc = run_acc.avg[0]
            # dice_wt = run_acc.avg[1]
            # dice_et = run_acc.avg[2]
            # print(
            #     "Val {}/{} {}/{}".format(epoch, max_epochs, idx, len(loader)),
            #     ", dice_tc:",
            #     dice_tc,
            #     ", dice_wt:",
            #     dice_wt,
            #     ", dice_et:",
            #     dice_et,
            #     ", time {:.2f}s".format(time.time() - start_time),
            # )
            # start_time = time.time()

    return run_acc.avg


def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
):
    print(time.ctime())
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, max_epochs):
        # print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
        )
        print(
            "Final training  {}/{}".format(epoch, max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    filename=f"run3_swin_ft_best.pt",
                    best_acc=val_acc_max,
                )
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )


start_epoch = 0

(
    val_acc_max,
    dices_tc,
    dices_wt,
    dices_et,
    dices_avg,
    loss_epochs,
    trains_epoch,
) = trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_func=dice_loss,
    acc_func=dice_acc,
    scheduler=scheduler,
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_sigmoid=post_sigmoid,
    post_pred=post_pred,
)

print("val_acc", val_acc_max)
print("dices_tc", dices_tc)
print("dices_wt", dices_wt)
print("dices_wt", dices_et)
print("dices_avg", dices_avg)
print("loss_epochs", loss_epochs)
print("trains_epochs", trains_epoch)

save_checkpoint(model, 1000, filename=f"run2_ours_ft_final.pt", best_acc=val_acc_max)
