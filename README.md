# APTUS-AI-powered-Patient-centric-acne-Treatment-via-a-Unified-skin-management-System

🚀 **Beta version now available for testing!**  

Scan the QR code below to access the demo:  

![QR Code](APTUS_Mini-program_QR_code.png)  

*Note: This is an early beta - features may be limited and subject to change.*

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

## 📝 License

This project is licensed under the [MIT License](LICENSE).
