# Object Detection on PASCAL VOC using Faster RCNN

This project implements an object detection model using Fast RCNN with PyTorch on the PASCAL VOC dataset. The model is trained to detect and classify 20 common object classes like animals, vehicles, household objects etc. It includes scripts for training, testing, and dataset processing.

## Training

To train a model, run:

```
python train.py --data_path /path/to/voc/data --epochs 50 --batch_size 4 --image_size 416
```

The main training script is `train.py`. It will train a Faster R-CNN model with a MobileNetV3 backbone on the PASCAL VOC dataset.

The key arguments are:

- `--data_path`: Path to the PASCAL VOC dataset 
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--image_size`: Input image size 

Training logs and checkpoints will be saved to `tensorboard/pascal_voc` and `trained_models` respectively by default.

## Testing

To test a trained model on an image, run:

```
python test.py --image test.jpg --checkpoint trained_models/best.pt
```

This will run inference on `test.jpg` using the model checkpoint at `trained_models/best.pt`, and save the prediction visualization to `prediction.jpg`.

## Dataset

The `dataset.py` script converts the PASCAL VOC dataset into a format suitable for PyTorch training. It handles resizing images to the desired input size, normalization, and converting annotations to labeled bounding boxes.

The customized `VOCDataset` class can be used with PyTorch `DataLoader` for easy batching and augmentation.

## Requirements

- PyTorch
- OpenCV
- torchvision
- torchmetrics

## References

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
 
