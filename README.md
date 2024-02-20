# Self-Supervised Modality-Agnostic Pre-Training of Swin Transformers

Paper Abstract:
> Unsupervised pre-training has emerged as a transformative paradigm, displaying remarkable advancements in various domains. However, the susceptibility to domain shift, where pre-training data distribution differs from fine-tuning, poses a significant obstacle. To address this, we augment the Swin Transformer to learn from different medical imaging modalities, enhancing downstream performance. Our model, dubbed SwinFUSE (Swin Multi-Modal Fusion for UnSupervised Enhancement), offers three key advantages: (i) it learns from both Computed Tomography (CT) and Magnetic Resonance Images (MRI) during pre-training, resulting in complementary feature representations; (ii) a domain-invariance module (DIM) that effectively highlights salient input regions, enhancing adaptability; (iii) exhibits remarkable generalizability, surpassing the confines of tasks it was initially pre-trained on.
Our experiments on two publicly available 3D segmentation datasets show a modest 1-2% performance trade-off compared to single-modality models, yet significant out-performance of up to 27% on out-of-distribution modality. This substantial improvement underscores our proposed approach's practical relevance and real-world applicability.


## Training

### Pre-training

- `scripts/main.py` and `scripts/train_job.sh` contains the script to pre-train the model
- Our proposed model with the DIM is defined as `SwinTransformerCoAttn` in `swin_unetr.py`
- This file exists in `monai.networks.nets` and is a modified version of the original `SwinTransformer` in `timm.models.swin_transformer`. The changes are made to include the DIM module and the multi-modal fusion module. Please replace this file in your MONAI installation with the provided file.
- Note: The script is designed to run on a single-node multi-gpu setup. Please modify the script to run on multi-node multi-gpu. Also, `torchrun` is preferred over `torch.distributed.launch`.

### Fine-training

- For BRATS'21 dataset, the fine-tuning script is provided in `brats_fine_tune.py` 
- For MSD dataset, the fine-tuning script is provided in `msd_fine_tune.py`

If you find this work useful, please cite our paper:

Talasila, Abhiroop; Maity, Maitreya; Priyakumar, U. Deva (2024): Self-Supervised Modality-Agnostic Pre-Training of Swin Transformers. In Proceedings of the 2024 IEEE 21st International Symposium on Biomedical Imaging (ISBI 2024), IEEE, 2024.