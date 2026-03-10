# APTUS-AI-powered-Patient-centric-acne-Treatment-via-a-Unified-skin-management-System

## 📱 Live Demo

Want to see our model in action? Try our mobile web demo:

👉 **[Click here to try APTUS](https://skin.beifuting.com/index)** 👈

*Best viewed on mobile devices. Simply scan three facial views (left, front, right) to see acne detection results.*

---

*Note: The demo link opens in your mobile browser - no installation required.*

## 📦 Installation
### Clone the Repository
```bash
git clone https://github.com/zzl-997/APTUS.git
cd APTUS
```

### Install Dependencies

```bash
conda create -n aptus_env python=3.8
conda activate aptus_env
pip install -r requirements.txt
```
## 🧠 Training
### Dataset Preparation

Place your dataset in the `data/` directory with the following structure:
```
data/
    APTUS/
        images/
            train/
                00001.png/
                00002.png/
            val/
                ......
                ......
            test/
                ......
                ......
        labels/
            train/
                00001.txt/
                00002.txt/
            val/
                ......
                ......
            test/
                ......
                ......
```

Additionally, you need to prepare a `.yaml` file that describes your dataset and training configuration. Below is an example of what this `.yaml` file looks like:

```yaml
path: /../data/APTUS  # dataset root dir
train: images/train  # train images (relative to 'path') 
val: images/val  # val images (relative to 'path') 
test: images/test  # test images (relative to 'path') 
# Classes
names:
  0: comedo
  1: papule
  2: pustule
  3: nodule
  4: ruptured
  5: cyst
  6: acne_scar
```

### Start Training

Run the training script with default settings:
```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 /../yolov9/segment/train.py \
    --workers 8 \
    --device 0,1,2,3,4,5,6,7 \
    --sync-bn \
    --batch 160 \
    --data /../yolov9/data/your_dataset.yaml \
    --img 640 \
    --cfg /../yolov9/models/segment/gelan-c-seg.yaml \
    --hyp /../yolov9/data/hyps/hyp.scratch-high.yaml \
    --no-overlap --epochs 40 \
    --close-mosaic 10 \
```

## 🔍 Inference
To run inference with a trained model:
```bash
python /../yolov9/segment/val.py \
    --workers 8 \
    --device 1 \
    --batch-size 120 \
    --data /../yolov9/data/your_dataset.yaml \
    --weights /trained_model_path \
    --img 640 \
    --iou-thres 0.5 \
    --max-det 50 \
    --task val \
    --verbose \
    --save-json \
```

# Multi-View Face Mesh Optimization - Configuration Guide

## 📋 Parameter Summary

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `is_match` | Enable/disable matching between views | `True` |
| `match_method` | Matching algorithm: 'hungarian' (optimal) or 'greedy' (fast) | `'hungarian'` |
| `distance_scale` | Scaling factor for matching distance threshold | `1.5` |
| `facemesh_confidence` | MediaPipe face detection confidence (0-1) | `0.5` |
| `landmark_numbers` | Number of facial landmarks (MediaPipe default: 468) | `468` |
| `class_threshold_strict` | High-confidence threshold per class | `{'class': 0.7}` |
| `class_threshold_loose` | Minimum confidence threshold per class | `{'class': 0.3}` |
| `float_max` | Large value for cost matrix infinity | `1e10` |

## 🚀 Quick Start Example

```python
config = {
    # Matching control
    'is_match': True,
    'match_method': 'hungarian',      # 'hungarian' or 'greedy'
    'distance_scale': 1.5,            # Higher = more matches
    
    # Face detection
    'facemesh_confidence': 0.5,       # MediaPipe confidence
    'landmark_numbers': 468,          # MediaPipe landmarks
    
    # Class thresholds (customize for your classes)
    'class_threshold_strict': {'lesion_name1': 0.7},   # Always show
    'class_threshold_loose': {'lesion_name1': 0.55},    # Minimum confidence
    'float_max': 1e10
}
```

## 📌 Key Points

- **Strict threshold**: Detections above this are always shown and used in strict matching
- **Loose threshold**: Detections below this are filtered out


## 📝 License

Dual-licensed:
- **Original matching & optimization code**: [MIT License](LICENSE.md)-This code is distributed as a separate, independent module that communicates with the detection module through data interfaces. It does not constitute a derivative work of the GPL-licensed code and therefore remains under the permissive MIT license.
- **YOLOv9-based detection module**: [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.html)-- This module is a derivative work of ([YOLOv9](https://github.com/WongKinYiu/yolov9)) and inherits its GPL-3.0 license.
The two components are designed as independent modules that can be used separately. When distributed together as a complete system, they form an "aggregate" as defined in GPL-3.0 Section 5, where including GPL-licensed code does not cause the license to apply to other independent parts.

Please ensure compliance with both licenses when using this project.
