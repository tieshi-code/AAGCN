# Quick Start Guide

## Prerequisites

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install torchlight:
```bash
cd torchlight
python setup.py install
cd ..
```

## Training

### Single Fold

```bash
python main.py recognition -c config/ASgcn/train_fold1.yaml
```

### All 5 Folds

```bash
for i in {1..5}; do
  python main.py recognition -c config/ASgcn/train_fold${i}.yaml
done
```

## Data Format

Your data should be organized as:
```
data/
  fold_1/
    train_label.pkl
    val_label.pkl
    *.npy (skeleton sequences)
  fold_2/
    ...
```

Each `.npy` file should contain skeleton data with shape `[C, T, V, M]`:
- C: channels (typically 3 for x, y, confidence)
- T: temporal frames (e.g., 300)
- V: number of joints (17 for COCO)
- M: number of persons (typically 1)

The `.pkl` files should contain lists of labels (integers 0-4 for 5 performance levels).

## Output

Training outputs will be saved in:
```
work_dir/recognition/last_amputee/ASgcn_fold1/
  epoch10_model.pt
  epoch20_model.pt
  ...
  log.txt
```

## Troubleshooting

- **Import errors**: Run from project root; add `.` to `PYTHONPATH`
- **CUDA errors**: Check that PyTorch is installed with CUDA support
- **Data errors**: Verify your `.npy` and `.pkl` files match the expected format
