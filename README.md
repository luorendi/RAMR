# RAMR: Reliability-Aware Modality Routing

This repository contains the training code for **RAMR**, a multimodal face anti-spoofing method based on reliability-aware modality routing. The code is organized to reproduce the main teacher-student training pipeline described in the paper:

**Multi-Modal Face Anti-Spoofing via Reliability-Aware Modality Routing**.

RAMR first trains a tri-modal teacher model with RGB, IR, and Depth inputs. The teacher uses a Reliability Gate to estimate sample-level modality reliability and a Depth Geometry Enhancement Module (DGEM) to strengthen depth-related geometric cues. A dual-modal RGB-IR student is then trained with Reliability-Guided Distillation (RGD), which transfers the teacher's depth-informed routing behavior to a model that does not require Depth input at inference time.

The experiments in the paper are conducted on the public CASIA-SURF, CeFA, and WMCA datasets. Due to dataset license restrictions, this repository does not redistribute any dataset files. Users should download the datasets from their official sources and organize them according to the expected directory structure.

No trained checkpoints, local paths, or hardware-specific records are included in this public package.

## Repository Structure

```text
.
|-- ramr/
|   |-- datasets.py      # paired RGB/IR/Depth dataset loader
|   |-- losses.py        # focal loss and RGD loss
|   |-- metrics.py       # FAS metrics: APCER, BPCER, ACER, AUC
|   |-- models.py        # RAMR-T and RAMR-S model definitions
|   `-- utils.py         # reproducibility, checkpoint, and JSON helpers
|-- train_teacher.py     # train RAMR-T
|-- train_student.py     # train RAMR-S with RGD
|-- configs/
|   `-- example.yaml
`-- requirements.txt
```

## Environment

```bash
conda create -n ramr python=3.10
conda activate ramr
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA or CPU environment if needed.

## Data Format

The dataset loader expects paired RGB, IR, and Depth samples under each split:

```text
<DATA_ROOT>/
  train/
    real/
      RGB/
      IR/
      Depth/
    fake/
      RGB/
      IR/
      Depth/
  val/
    real/
      RGB/
      IR/
      Depth/
    fake/
      RGB/
      IR/
      Depth/
  test/
    real/
      RGB/
      IR/
      Depth/
    fake/
      RGB/
      IR/
      Depth/
```

File names are paired by normalized stem. Nested subject or protocol folders are supported as long as the relative folder and file stem are consistent across the three modalities.

Labels follow the common FAS setting:

- `real`: bona fide samples, label `0`
- `fake`: presentation attack samples, label `1`

Input sizes:

- RGB input: `256 x 256`
- IR input: `64 x 64`
- Depth input: `64 x 64`

## Training the Teacher

```bash
python train_teacher.py \
  --train_root <DATA_ROOT>/train \
  --val_root <DATA_ROOT>/val \
  --output_dir outputs/teacher \
  --epochs 40 \
  --batch_size 32
```

Optional switches:

```bash
--disable_gate   # train without the Reliability Gate
--disable_dgem   # train without DGEM
--dual_pretrained_ckpt <DUAL_CHECKPOINT>  # optional RGB-IR initialization
```

The script saves:

```text
outputs/teacher/best.pt
outputs/teacher/last.pt
outputs/teacher/history.json
```

## Training the Student

```bash
python train_student.py \
  --train_root <DATA_ROOT>/train \
  --val_root <DATA_ROOT>/val \
  --teacher_ckpt outputs/teacher/best.pt \
  --output_dir outputs/student \
  --epochs 40 \
  --batch_size 32
```

RAMR-S uses only RGB and IR during inference. Its RGB-IR fusion uses bidirectional multi-head cross-attention over four learned latent sub-tokens per modality. During training, the frozen teacher provides depth-informed reliability guidance through RGD.

An optional dual-modal checkpoint can be provided with `--dual_pretrained_ckpt <DUAL_CHECKPOINT>` when reproducing experiments that start from an RGB-IR initialization.

## Notes

- The default optimizer is AdamW.
- The default learning rate is `1e-4`.
- The default learning rate schedule is cosine annealing.
- Teacher training uses cross-entropy loss. Student training uses focal loss together with RGD.
- Early stopping is based on validation ACER.
- Checkpoints and datasets are not distributed in this repository.
- Dataset paths and protocol splits should be prepared by users according to their own experimental setup.
