from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor, \
    FasterRCNN_MobileNet_V3_Large_FPN_Weights
import torch
import cv2
import argparse
import numpy as np
import time


def collate_fn(batch):
    all_images = []
    all_labels = []
    for image, label in batch:
        all_images.append(image)
        all_labels.append(label)
    return all_images, all_labels


def get_args():
    parser = argparse.ArgumentParser(description="Animals classifier")
    parser.add_argument("-i", "--image", type=str, default="test_images/1.jpg", help="path to test image")
    parser.add_argument("-s", "--size", default=416, type=int)
    parser.add_argument("-t", "--conf_threshold", default=0.5, type=float)
    parser.add_argument("-c", "--checkpoint", type=str, default="trained_models/last.pt",
                        help="path to model checkpoint file")

    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                  'train', 'tvmonitor']
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features,
                                                      num_classes=len(categories))
    model.to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ori_image = cv2.imread(args.image)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    image = cv2.resize(image, (args.size, args.size))
    image = image / 255.
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    image = [torch.from_numpy(np.transpose(image, (2, 0, 1))).to(device).float()]
    with torch.no_grad():
        predictions = model(image)
    for box, score, label in zip(predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"]):
        if score > args.conf_threshold:
            xmin, ymin, xmax, ymax = box
            xmin = int(xmin / args.size * width)
            ymin = int(ymin / args.size * height)
            xmax = int(xmax / args.size * width)
            ymax = int(ymax / args.size * height)
            cv2.rectangle(ori_image, (xmin, ymin), (xmax, ymax), (128, 0, 128), 2)
            cv2.putText(ori_image, categories[label] + "{:0.2f}".format(score), (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 0), 1)
    cv2.imwrite("prediction.jpg", ori_image)
    # cv2.imshow("test", ori_image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    args = get_args()
    test(args)
