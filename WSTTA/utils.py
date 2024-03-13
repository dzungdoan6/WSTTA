import torch
from detectron2.data import transforms as T
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

# create weak labels from true ground truth
# This is to simulate a human operator who provides the weak labels
def create_weak_label(inputs):
    file_name = inputs[0]['file_name'];
    height = inputs[0]['height'];
    width = inputs[0]['width'];
    image_id = inputs[0]['image_id'];
    orig_image = inputs[0]['image'];
    
    gt_classes = inputs[0]['instances'].gt_classes;
    weak_classes = torch.unique(gt_classes, sorted = True);

    
    return [{'file_name': file_name,
             'height': height,
             'width': width,
             'image_id': image_id,
             'image': orig_image, 
             'weak_classes': weak_classes}];
    
    
# create pseudo labels from predictions and weak labels
# psd_thr is the prob threshold to create pseudo labels
def create_pseudo_labels(inputs, preds, psd_thr=0.8):
    assert inputs[0]['height'] == preds[0]['instances'].image_size[0]
    assert inputs[0]['width'] == preds[0]['instances'].image_size[1]
    
    # unwrap information
    file_name = inputs[0]['file_name'];
    height = inputs[0]['height'];
    width = inputs[0]['width'];
    image_id = inputs[0]['image_id'];
    orig_image = inputs[0]['image'];
    weak_labels = inputs[0]['weak_classes'];

    pred_boxes = preds[0]['instances'].pred_boxes.tensor.cpu().detach();
    pred_scores = preds[0]['instances'].scores.cpu().detach();
    pred_classes = preds[0]['instances'].pred_classes.cpu().detach();

    pseudo_classes = list();
    pseudo_boxes = list();
    
    # for each category from weak labels, we check if it exists in the prediction
    # if it exists, find bounding boxes with scores > threshold
    for cls in weak_labels:
        indices = (pred_classes == cls).nonzero(as_tuple = True)[0];
        if indices.nelement() > 0:
            
            scores = pred_scores[indices]
            boxes = pred_boxes[indices,:]
            k = find_ids_greater_than_threshold(scores=scores, thr=psd_thr);

            if len(k) > 0:
                pseudo_classes.append(cls.repeat(len(k)));
                pseudo_boxes.append(Boxes(boxes[k,:]));

    assert len(pseudo_classes) == len(pseudo_boxes);
    if len(pseudo_classes) > 0:            
        pseudo_classes = torch.cat(pseudo_classes, dim=0);
        pseudo_boxes = Boxes.cat(pseudo_boxes);

    # wrap pseudo labels to return
    instances = Instances(
                          image_size=(height, width), \
                          gt_boxes = pseudo_boxes, \
                          gt_classes = pseudo_classes
                          );
    return [{'file_name': file_name,
             'height': height,
             'width': width,
             'image_id': image_id,
             'image': orig_image, 
             'weak_classes': weak_labels,
             'instances': instances}];
    
    
def find_ids_greater_than_threshold(scores, thr):
    ids = (scores >= thr).nonzero();
    if len(ids) > 0:
        return ids[0] if len(ids) == 1 else ids.squeeze();
    return [];