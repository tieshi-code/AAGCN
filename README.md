# AAGCN (Amputee-Aware Graph Convolutional Network)

AS-GCN with amputee attention for 5-level gait performance classification. Uses structural graph (fixed anatomy) + actional graph (self-attention) and channel/joint attention with amputation prior.

- Structural + actional graph convolution; amputee leg attention (SE + joint mask)
- 5-fold configs; 17 COCO keypoints

## Model Architecture

`net/aagcn.py`: Structural + Actional graph convolution fusion per paper formula, followed by ALAM (`AmputeeLimbAttention`) using the amputation prior mask `m`, then GAP+FC(5).


## Installation

### Prerequisites

- Python 3.6+
- PyTorch 1.7+ (tested with CUDA 10.2+)
- NumPy, OpenCV, tqdm

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd aagcn

# Install dependencies
pip install -r requirements.txt

# Install torchlight (logging utility)
cd torchlight
python setup.py install
cd ..
```

## Data Preparation

No training data included. Prepare data as:

- **Input**: `.npy` files containing skeleton sequences (shape: `[C, T, V, M]` where C=channels, T=frames, V=joints, M=persons)
- **Labels**: `.pkl` files with class labels (0-4 for 5 performance levels)
- **Amputee Types**: Metadata indicating left/right leg amputation (used for attention masking)

The data should be organized in 5-fold cross-validation splits:
```
data/
  fold_1/
    train_label.pkl
    val_label.pkl
    (npy files)
  fold_2/
    ...
```

See the original ST-GCN repository or `feeder/feeder_kinetics.py` for data format details.

## Training

### Single Fold Training

Train a single fold (e.g., fold 1):

```bash
python main.py recognition -c config/ASgcn/train_fold1.yaml
```

### 5-Fold Cross-Validation

Train all 5 folds sequentially:

```bash
for i in {1..5}; do
  python main.py recognition -c config/ASgcn/train_fold${i}.yaml
done
```

### Configuration

Key hyperparameters in `config/ASgcn/train_fold*.yaml`:

- `model: net.aagcn.AAGCN` - Amputee-Aware Graph Convolutional Network (AAGCN)
- `lambda_action: 0.5` - Weight for actional branch fusion
- `attention_weight: 2.0` - Amplification factor for amputee attention
- `base_lr: 0.0003` - Learning rate
- `batch_size: 8` - Batch size
- `num_epoch: 120` - Training epochs
- `scheduler: CosineAnnealingLR` - Learning rate scheduler

## Model Files

- `net/aagcn.py`: Strict AAGCN implementation
  - `AAGCNBlock`: Structural (fixed Â_st) + Actional (adaptive Â_at) fusion via paper formula
  - `AmputeeLimbAttention`: ALAM (channel attention + prior-guided joint attention using m)
  - `AAGCN`: Main 10-layer backbone + GAP + FC(5)
- `net/utils/`: Graph utilities (`graph.py`, `tgcn.py`)


## Directory Structure

```
aagcn/
├── net/                    # Model definitions
│   └── aagcn.py            # Strict AAGCN implementation (incl. ALAM)
│   └── utils/             # Graph utilities
├── config/                # Training configurations
│   └── ASgcn/             # 5-fold configs
├── feeder/                # Data loaders
├── processor/             # Training processor
├── torchlight/            # Logging utilities
├── main.py                # Training entry point
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Citation

If you use this code, please cite:

```bibtex
@article{yan2018spatial,
  title={Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  journal={AAAI},
  year={2018}
}

@inproceedings{li2019actional,
  title={Actional-Structural Graph Convolutional Networks for Skeleton-based Action Recognition},
  author={Li, Maosen and Chen, Siheng and Chen, Xu and Zhang, Ya and Wang, Yanfeng and Tian, Qi},
  booktitle={CVPR},
  year={2019}
}
```

## License

This project follows the same license as the original ST-GCN repository. See `LICENSE` for details.

## Acknowledgments

- Original ST-GCN: [https://github.com/yysijie/st-gcn](https://github.com/yysijie/st-gcn)
- AS-GCN inspiration: [Actional-Structural Graph Convolutional Networks](https://arxiv.org/abs/1904.12659)
- AlphaPose for pose estimation: [https://github.com/MVIG-SJTU/AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)

## Contact

Open an issue on GitHub.
