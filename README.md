<!-- Animated Header -->
<img src="https://balaboom123-capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Two-Step%20Multi-Biometric%20Auth&fontSize=38&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Face%20%2B%20Voice%20Recognition%20Authentication%20System&descAlignY=52&descSize=18"/>

<div align="center">

![IET](https://img.shields.io/badge/IET_ICETA-Published_2024-success?style=flat) &nbsp;
![arXiv](https://img.shields.io/badge/arXiv-2601.06218-b31b1b?style=flat) &nbsp;

</div>

<p align="center">
A two-step biometric authentication system combining <strong>face recognition</strong> (VGG16 + MTCNN) and <strong>voice recognition</strong> (ResNet + Triplet Loss) for robust identity verification.
</p>

<!-- Quick Links -->
<div align="center">
  <a href="#-key-features"><img src="https://img.shields.io/badge/Key_Features-4285F4?style=flat-square" alt="Key Features"/></a>
  <a href="#-architecture"><img src="https://img.shields.io/badge/Architecture-34A853?style=flat-square" alt="Architecture"/></a>
  <a href="#-getting-started"><img src="https://img.shields.io/badge/Getting_Started-EA4335?style=flat-square" alt="Getting Started"/></a>
  <a href="#-results"><img src="https://img.shields.io/badge/Results-FBBC05?style=flat-square" alt="Results"/></a>
</div>

<br/>

---

## Key Features

<div align="center">
<table>
<tr>
<td width="50%">

### 👤 Face Recognition
Fine-tuned **VGG16** with MTCNN face detection, data augmentation, and two-stage training (frozen → unfrozen layers)

### 🎙️ Voice Preprocessing
**VAD** (Voice Activity Detection) and **Fbank** feature extraction with FLAC-to-WAV conversion for LibriSpeech

### ⚡ Real-Time Inference
Webcam-based face capture and live speaker verification with confidence thresholds

</td>
<td width="50%">

### 🔊 Voice Recognition
Custom **ResNet** architecture with triplet loss and cosine similarity for speaker verification

### 🔄 Two-Stage Training
Random batch pre-training followed by selected batch refinement for optimized convergence

### 📊 Comprehensive Evaluation
Accuracy, EER, precision, recall, and F-measure metrics with training curve visualization

</td>
</tr>
</table>
</div>

---

## Architecture

### System Prototype

<div align="center">

![workflow](https://github.com/NCUE-EE-AIAL/Two-step-Authentication-Multi-biometric-System/blob/main/doc/Flow_diagram.png)

</div>

### Face Recognition Model

<div align="center">

![face model](https://github.com/NCUE-EE-AIAL/Two-step-Authentication-Multi-biometric-System/blob/main/doc/graph_hr.png)

</div>

### Voice Recognition Model

```mermaid
graph TD
    subgraph CNNs
        A2[Input Layer] --> B2[ResNet Block: filter=64]
        B2 --> C2[ResNet Block: filter=128]
        C2 --> D2[ResNet Block: filter=256]
        D2 --> G2[ResNet Block: filter=512]

        G2 --> N2[Reshape & Mean]
        N2 --> P2[Dense 512]
        P2 --> Q2[Output Layer]
    end

    subgraph ResNet block
        A3[Input Tensor] --> B3[Conv2D Layer: kernel_size=5]
        B3 --> C3[BatchNormalization]
        C3 --> D3[Clipped ReLU]
        D3 --> E3[Identity Block * 3]
        E3 --> F3[Output Tensor]
    end

    subgraph Identity Block
        A[Input Tensor] --> B[Conv2D Layer: kernel_size=1]
        A[Input Tensor] --> J[+]
        B --> C[BatchNorm -> Clipped ReLU]
        C --> E[Conv2D Layer: kernel_size=3]
        E --> F[BatchNorm -> Clipped ReLU]
        F --> H[Conv2D Layer: kernel_size=1]
        H --> I[BatchNormalization]
        I --> J
        J --> K[Clipped ReLU]
        K --> L[Output Tensor]
    end
```

---

## Dataset

| Modality | Source | Details |
|----------|--------|---------|
| **Face** | Custom dataset | Collected from EE's class students (see `Dataset/` folder) |
| **Voice** | [LibriSpeech](https://www.openslr.org/12) | `train-clean-360` for training, `test-clean` for evaluation |

---

## Getting Started

### Project Structure

```
.
├── image_preprocessing.py   # MTCNN face detection & cropping
├── train_face.ipynb         # VGG16 fine-tuning for face recognition
├── test_face.py             # Webcam-based face verification
├── voice_preprocessing.py   # FLAC→WAV, VAD, Fbank extraction
├── train_voice.py           # Two-stage voice model training
├── test_voice.py            # Speaker verification evaluation
├── src/
│   ├── models.py            # Model architectures
│   ├── triplet_loss.py      # Triplet loss implementation
│   ├── random_batch.py      # Random batch sampling
│   ├── select_batch.py      # Selected batch sampling
│   ├── silence_detector.py  # Audio silence detection
│   ├── constants.py         # Configuration constants
│   └── utils.py             # Utility functions
├── eval/
│   └── eval_metrics.py      # Evaluation metrics (EER, F-measure, etc.)
├── Dataset/                 # Face image dataset
├── doc/                     # Architecture diagrams & result graphs
└── checkpoints_sample/      # Sample model checkpoints
```

### Usage

#### 1. Face Recognition Pipeline

```bash
# Step 1: Preprocess face images (detect & crop faces)
python image_preprocessing.py

# Step 2: Train the face recognition model (open in Jupyter)
jupyter notebook train_face.ipynb

# Step 3: Test with webcam
python test_face.py
```

#### 2. Voice Recognition Pipeline

```bash
# Step 1: Preprocess voice data (FLAC→WAV, VAD, Fbank)
python voice_preprocessing.py

# Step 2: Train the voice recognition model
python train_voice.py

# Step 3: Evaluate speaker verification
python test_voice.py
```

### Script Details

| Script | Description |
|--------|-------------|
| `image_preprocessing.py` | Detects and crops faces from images using MTCNN, reads paths from CSV, saves cropped faces maintaining directory structure |
| `train_face.ipynb` | Fine-tunes VGG16: freezes conv layers → trains custom FC layers → unfreezes and fine-tunes with lower learning rate |
| `test_face.py` | Captures webcam frame, crops face via MTCNN, runs model prediction, outputs match label if confidence exceeds threshold |
| `voice_preprocessing.py` | Processes LibriSpeech files (speaker-name format), converts FLAC→WAV, applies VAD and Fbank feature extraction |
| `train_voice.py` | Two-stage training: random batches for initial convergence, then selected batches for refinement, with per-epoch validation |
| `test_voice.py` | Evaluates speaker verification using triplet loss model with cosine similarity, reports accuracy, EER, precision, recall, F-measure |

---

## Results

<div align="center">

| System | Accuracy | Equal Error Rate | Precision | Recall |
|--------|----------|------------------|-----------|--------|
| **Face Recognition** | **95.135%** | - | 96.317% | 95.153% |
| **Voice Recognition** | **99.1%** | 3.456% | 86.48% | 88.65% |

</div>

<div align="center">

![face result](https://github.com/NCUE-EE-AIAL/Two-step-Authentication-Multi-biometric-System/blob/main/doc/training_graph.png)

*Training and validation curves for face recognition: (a) Accuracy (b) Loss*

![voice result](https://github.com/NCUE-EE-AIAL/Two-step-Authentication-Multi-biometric-System/blob/main/doc/training_graph_voice.png)

*Training and validation curves for voice recognition: (a) EER (b) Loss*

</div>

---

## Acknowledgement

This research is supported by **TEEP** (Taiwan Experience Education Program) at **National Changhua University of Education**.

---

<!-- Animated Footer -->
<img src="https://balaboom123-capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer"/>
