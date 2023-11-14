from yolo_utils import load_image, show_img, predict
import torch
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='region gan')
  parser.add_argument('-img_path', '--path', help='image path', required=True)
  parser.add_argument('-m', '--model', help='model weights', required=True)
  parser.add_argument('-conf', '--conf', help='NMS confidence threshold', required=True)
  parser.add_argument('-iou', '--iou', help='IoU threshold', required=True)
  parser.add_argument('-size', '--size', help='image size', required=True)

  args = parser.parse_args()

  path = args.path
  ckpt_path = args.model
  conf      = float(args.conf)
  iou       = float(args.iou)
  size = int(args.size)

  model = torch.hub.load('yolov5','custom',path=ckpt_path,source='local',force_reload=True)
  model.conf = conf  
  model.iou  = iou  
  model.multi_label=False
  model.agnostic = True

  img=load_image(path)
  bboxes, confis,names,labels = predict(model, img, size=size)
  result=show_img(img, bboxes=bboxes,names=names,labels=labels,confs=None, bbox_format='voc_pascal',show_classes = True)
  result.save("output_image.png")
