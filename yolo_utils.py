import cv2
import random
from PIL import Image

def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)

def plot_one_box(x, img, color=None, label=None, conf=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        label = f"{label}:{conf:.2f}" if conf is not None else label
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_bboxes(img, bboxes, classes, class_ids, confs, colors = None, show_classes = None, bbox_format = 'yolo', class_name = False, line_thickness = 2):

    image = img.copy()
    
    colors =     colors = {
    0: (0, 255, 0),   
    1: (255, 0, 0),   
    2: (255, 182, 193),   
    } 

    if bbox_format == 'yolo':

        for idx in range(len(bboxes)):

            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            conf = confs[idx] if confs is not None else None
            color = colors.get(cls_id)


            x1 = round(float(bbox[0])*image.shape[1])
            y1 = round(float(bbox[1])*image.shape[0])
            w  = round(float(bbox[2])*image.shape[1]/2) 
            h  = round(float(bbox[3])*image.shape[0]/2)

            voc_bbox = (x1-w, y1-h, x1+w, y1+h)
            plot_one_box(voc_bbox,
                             image,
                             color = color,
                             conf=conf,
                             label = cls if show_classes else str(cls_id),
                             line_thickness = line_thickness)

    elif bbox_format == 'coco':

        for idx in range(len(bboxes)):

            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            conf = confs[idx] if confs is not None else None
            color = colors.get(cls_id)
            

           
            x1 = int(round(bbox[0]))
            y1 = int(round(bbox[1]))
            w  = int(round(bbox[2]))
            h  = int(round(bbox[3]))

            voc_bbox = (x1, y1, x1+w, y1+h)
            plot_one_box(voc_bbox,
                             image,
                             color = color,
                             label = cls if show_classes else str(cls_id),
                             line_thickness = line_thickness)

    elif bbox_format == 'voc_pascal':

        for idx in range(len(bboxes)):

            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            conf = confs[idx] if confs is not None else None
            color = colors.get(cls_id)

            
            x1 = int(round(bbox[0]))
            y1 = int(round(bbox[1]))
            x2 = int(round(bbox[2]))
            y2 = int(round(bbox[3]))
            voc_bbox = (x1, y1, x2, y2)
            plot_one_box(voc_bbox,
                             image,
                             color = color,
                             conf=conf,
                             label = cls if show_classes else str(cls_id),
                             line_thickness = line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image

def show_img(img, bboxes,names,labels,confs=None, bbox_format='voc_pascal',show_classes = None):
    img=draw_bboxes(img = img,
                           bboxes = bboxes,
                           classes = names,
                           class_ids = labels,
                           confs=confs,
                           class_name = True,
                           colors = None,
                           bbox_format = bbox_format,
                           show_classes = show_classes,
                           line_thickness = 2)
    return Image.fromarray(img)    

def predict(model, img, size=640, augment=False):
    height, width = img.shape[:2]
    results = model(img, size=size, augment=augment)  # custom inference siz
    preds   = results.pandas().xyxy[0]
    bboxes  = preds[['xmin','ymin','xmax','ymax']].values
    confs   = preds.confidence.values
    class_id   =preds['class'].values
    name=preds.name.values

    return bboxes, confs, name,class_id

