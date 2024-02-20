import json
import os
import time
import sys
import matplotlib.pyplot as plt
import torch
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.losses import ContrastiveLoss
from monai.networks.nets import ViTAutoEnc
from monai.transforms import (
    Compose,
    CopyItemsd,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    OneOf,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    RandRotated,
    RandZoomd,
)
from monai.utils import first, set_determinism
from sklearn.model_selection import train_test_split
from torch.nn import L1Loss
from tqdm import tqdm

BRAIN_ROOT = "/raid/deva/Task1/brain/"
LOG_ROOT = "/raid/deva/logs"
set_determinism(seed=123)
CUDA_ID = sys.argv[1]

if CUDA_ID is None:
    print("Please provide CUDA ID")
    sys.exit()

def move_files_to_root():
    for x in os.listdir(BRAIN_ROOT):
        if "overview" not in x:
            for itm in os.listdir(BRAIN_ROOT + x):
                src = os.path.join(BRAIN_ROOT + x, itm)
                dest = os.path.join(BRAIN_ROOT, x + "_" + itm)
                os.rename(src, dest)

def get_files_list():
    masks, mris, cts = [], [], []

    for x in os.listdir(BRAIN_ROOT):
        if "mask" in x:
            masks.append(os.path.join(BRAIN_ROOT, x))
        elif "mr" in x:
            mris.append(os.path.join(BRAIN_ROOT, x))
        elif "ct" in x:
            cts.append(os.path.join(BRAIN_ROOT, x))

    masks.sort()
    mris.sort()
    cts.sort()
    return masks, mris, cts

def create_samples(inp_list, ids):
    samples = []
    for x in inp_list:
        if x.split("_")[0] in ids:
            samples.append({"image": x})
    return samples


masks, mris, cts = get_files_list()

ids = [x.split("_")[0] for x in masks]
train_ids, test_ids = train_test_split(ids, test_size=0.1, random_state=123)

train_masks = create_samples(masks, train_ids)
train_mris = create_samples(mris, train_ids)
train_cts = create_samples(cts, train_ids)

test_masks = create_samples(masks, test_ids)
test_mris = create_samples(mris, test_ids)
test_cts = create_samples(cts, test_ids)

# Define Training Transforms
train_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear")),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(128, 128, 128)),
        RandSpatialCropSamplesd(
            keys=["image"], roi_size=(128, 128, 128), random_size=False, num_samples=2
        ),
        RandRotated(
            keys=["image"], range_x=90, range_y=90, range_z=90, prob=0.5, keep_size=True
        ),
        RandZoomd(keys=["image"], min_zoom=0.8, max_zoom=1.2, prob=0.4, keep_size=True),
        CopyItemsd(
            keys=["image"],
            times=2,
            names=["gt_image", "image_2"],
            allow_missing_keys=False,
        ),
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image"],
                    prob=1.0,
                    holes=6,
                    spatial_size=5,
                    dropout_holes=True,
                    max_spatial_size=32,
                ),
                RandCoarseDropoutd(
                    keys=["image"],
                    prob=1.0,
                    holes=6,
                    spatial_size=20,
                    dropout_holes=False,
                    max_spatial_size=64,
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
        # Please note that that if image, image_2 are called via the same transform call because of the determinism
        # they will get augmented the exact same way which is not the required case here, hence two calls are made
        OneOf(
            transforms=[
                RandCoarseDropoutd(
                    keys=["image_2"],
                    prob=1.0,
                    holes=6,
                    spatial_size=5,
                    dropout_holes=True,
                    max_spatial_size=32,
                ),
                RandCoarseDropoutd(
                    keys=["image_2"],
                    prob=1.0,
                    holes=6,
                    spatial_size=20,
                    dropout_holes=False,
                    max_spatial_size=64,
                ),
            ]
        ),
        RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
    ]
)


import torch.nn as nn
from collections.abc import Sequence
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv
from monai.utils import ensure_tuple_rep
from monai.networks.blocks.selfattention import SABlock


class CustomViTAutoEnc(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        out_channels: int = 1,
        deconv_chns: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.spatial_dims = spatial_dims

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=384,  ### hidden size is halved because we have two inputs
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=self.spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

        new_patch_size = [4] * self.spatial_dims
        conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        # self.conv3d_transpose* is to be compatible with existing 3d model weights.
        self.conv3d_transpose = conv_trans(
            hidden_size, deconv_chns, kernel_size=new_patch_size, stride=new_patch_size
        )
        self.conv3d_transpose_1 = conv_trans(
            in_channels=deconv_chns,
            out_channels=out_channels,
            kernel_size=new_patch_size,
            stride=new_patch_size,
        )

        #############################
        self.norm1 = nn.LayerNorm(hidden_size)
        self.patch_attn = SABlock(
            hidden_size, num_heads, dropout_rate, qkv_bias, save_attn
        )

    def forward(self, x1, x2):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        spatial_size = x1.shape[2:]

        x1 = self.patch_embedding(x1)
        x2 = self.patch_embedding(x2)
        x = torch.cat([x1, x2], dim=2)

        # self attn
        x = x + self.patch_attn(self.norm1(x))

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)

        # for blk in self.blocks:
        #     if i % 2 == 0:
        #         x1 = blk(x1)
        #     else:
        #         x2 = blk(x2)
        #     hidden_states_out.append(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)
        return x, hidden_states_out


# Define Network ViT backbone & Loss & Optimizer
model = CustomViTAutoEnc(
    in_channels=1,
    img_size=(128, 128, 128),
    patch_size=(16, 16, 16),
    pos_embed="conv",
    hidden_size=768,
    mlp_dim=2048,  ### reduced
)

device = torch.device(f"cuda:{CUDA_ID}")
model = model.to(device)

# Define Hyper-paramters for training loop
max_epochs = 500
val_interval = 2
batch_size = 4
lr = 1e-4
epoch_loss_values = []
step_loss_values = []
epoch_cl_loss_values = []
epoch_recon_loss_values = []
val_loss_values = []
best_val_loss = 1000.0

recon_loss = L1Loss()
contrastive_loss = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


train_ds_mri = Dataset(data=train_mris, transform=train_transforms)
train_loader_mri = DataLoader(
    train_ds_mri, batch_size=batch_size, shuffle=False, num_workers=4
)

train_ds_ct = Dataset(data=train_cts, transform=train_transforms)
train_loader_ct = DataLoader(
    train_ds_ct, batch_size=batch_size, shuffle=False, num_workers=4
)

val_ds_mri = Dataset(data=test_mris, transform=train_transforms)
val_loader_mri = DataLoader(
    val_ds_mri, batch_size=batch_size, shuffle=False, num_workers=4
)

val_ds_ct = Dataset(data=test_cts, transform=train_transforms)
val_loader_ct = DataLoader(
    val_ds_ct, batch_size=batch_size, shuffle=False, num_workers=4
)

pbar = tqdm(range(max_epochs))
for epoch in pbar:
    pbar.set_description(f"epoch {epoch + 1}/{max_epochs}")
    pbar.refresh()
    model.train()
    epoch_loss = 0
    epoch_cl_loss = 0
    epoch_recon_loss = 0
    step = 0

    for mri_, ct_ in zip(train_loader_mri, train_loader_ct):
        step += 1
        start_time = time.time()
        mri_inputs, mri_inputs_2, mri_gt_input = (
            mri_["image"].to(device),
            mri_["image_2"].to(device),
            mri_["gt_image"].to(device),
        )

        ct_inputs, ct_inputs_2, ct_gt_input = (
            ct_["image"].to(device),
            ct_["image_2"].to(device),
            ct_["gt_image"].to(device),
        )

        optimizer.zero_grad()
        outputs_v1, hidden_v1 = model(mri_inputs, ct_inputs)
        outputs_v2, hidden_v2 = model(mri_inputs_2, ct_inputs_2)

        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

        r_loss_mri = recon_loss(outputs_v1, mri_gt_input)
        r_loss_ct = recon_loss(outputs_v1, ct_gt_input)
        cl_loss = contrastive_loss(flat_out_v1, flat_out_v2)

        # Adjust the CL loss by Recon Loss
        r_loss = r_loss_mri + r_loss_ct
        total_loss = r_loss + cl_loss * r_loss

        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
        step_loss_values.append(total_loss.item())

        # CL & Recon Loss Storage of Value
        epoch_cl_loss += cl_loss.item()
        epoch_recon_loss += r_loss.item()

        end_time = time.time()
        # print(
        #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
        #     f"train_loss: {total_loss.item():.4f}, "
        #     f"time taken: {end_time-start_time}s"
        # )

    epoch_loss /= step
    epoch_cl_loss /= step
    epoch_recon_loss /= step

    epoch_loss_values.append(epoch_loss)
    epoch_cl_loss_values.append(epoch_cl_loss)
    epoch_recon_loss_values.append(epoch_recon_loss)
    pbar.set_description(f"epoch {epoch + 1}/{max_epochs} | loss: {epoch_loss:.4f}")
    pbar.refresh()

    if epoch % val_interval == 0:
        total_val_loss = 0
        val_step = 0
        model.eval()
        for mri_, ct_ in zip(val_loader_mri, val_loader_ct):
            val_step += 1
            start_time = time.time()

            mri_inputs, mri_gt_input = (
                mri_["image"].to(device),
                mri_["gt_image"].to(device),
            )

            ct_inputs, ct_gt_input = (
                ct_["image"].to(device),
                ct_["gt_image"].to(device),
            )

            outputs, outputs_v2 = model(mri_inputs, ct_inputs)
            r_loss_mri = recon_loss(outputs, mri_gt_input)
            r_loss_ct = recon_loss(outputs, ct_gt_input)

            val_loss = r_loss_mri + r_loss_ct
            total_val_loss += val_loss.item()
            end_time = time.time()

        total_val_loss /= val_step
        val_loss_values.append(total_val_loss)
        print(
            f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}, "
            f"time taken: {end_time-start_time}s"
        )

        if total_val_loss < best_val_loss:
            print(
                f"Saving new model based on validation loss {total_val_loss:.4f} epoch {epoch + 1}"
            )
            best_val_loss = total_val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(LOG_ROOT, "best_model.pt"))

        plt.figure(1, figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.plot(epoch_loss_values)
        plt.grid()
        plt.title("Training Loss")

        plt.subplot(2, 2, 2)
        plt.plot(val_loss_values)
        plt.grid()
        plt.title("Validation Loss")

        plt.subplot(2, 2, 3)
        plt.plot(epoch_cl_loss_values)
        plt.grid()
        plt.title("Training Contrastive Loss")

        plt.subplot(2, 2, 4)
        plt.plot(epoch_recon_loss_values)
        plt.grid()
        plt.title("Training Recon Loss")

        plt.savefig(os.path.join(LOG_ROOT, "loss_plots.png"))
        plt.close(1)

print("Done")
print(f"Saving new model based on validation loss {total_val_loss:.4f}")
checkpoint = {
    "epoch": max_epochs,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
torch.save(checkpoint, os.path.join(LOG_ROOT, "final_model.pt"))