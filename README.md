
<h2 align="center">
  NTIRE 2026 CROSS-DOMAIN FEW-SHOT OBJECT DETECTION (CDFSOD) CHALLENGE
</h2>



## 1. Datasets


### CD-FSOD

mkdir dataset

Download CD-FSOD datasets and organize as:

```shell
dataset/CDFSOD/
    ├── dataset1/...
    ├── datset2/...
    └── datset3/...
```

---

## 2. Quick Start

### Environment Setup

```bash
conda env create -f fsod.yml
conda activate FSODVFM
```

### UPN Installation

```bash
conda install -c conda-forge gcc=9.5.0 gxx=9.5.0 ninja -y
cd chatrex/upn/ops
pip install -v -e .
```

### Checkpoints

* For UPN, downdload "upn_large.pth" from https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth to ./checkpoints
* For RADIOv4, download "c-radio_v4-h_half.pth.tar" from https://huggingface.co/nvidia/C-RADIOv4-H/tree/main to ./checkpoints
* For SAM2, download "sam2.1_hiera_large.pt" from https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt to ./checkpoints
---

## 3. Usage

mkdir results

sh run_scripts/run_cdfsod_dataset1_1shot.sh

sh run_scripts/run_cdfsod_dataset1_5shot.sh

sh run_scripts/run_cdfsod_dataset1_10shot.sh

sh run_scripts/run_cdfsod_dataset2.sh

sh run_scripts/run_cdfsod_dataset3.sh

result json will be saved in ./results

---

## 4. Acknowledgement

Our work builds upon excellent open-source projects including
[FSOD-VFM](https://github.com/Intellindust-AI-Lab/FSOD-VFM)
[No-Time-To-Train](https://github.com/miquel-espinosa/no-time-to-train),
[SAM2](https://github.com/facebookresearch/sam2/tree/main),
[ChatRex](https://github.com/IDEA-Research/ChatRex),
[DINOv2](https://github.com/facebookresearch/dinov2), and
[RADIOv4](https://github.com/NVlabs/RADIO/tree/main).
We sincerely thank their authors for their contributions to the community.



