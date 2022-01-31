# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import copy
from collections import defaultdict
from pathlib import Path

import torch
import torch.utils.data
from transformers import RobertaTokenizerFast

import util.dist as dist
from util.box_ops import generalized_box_iou

import numpy as np
from .coco import ModulatedDetection, make_coco_transforms
import pycocotools.mask as mask_util


class RefExpDetection(ModulatedDetection):
    pass


class RefExpEvaluator(object):
    def __init__(self, refexp_gt, iou_types, k=(1, 5, 10), thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        refexp_gt = copy.deepcopy(refexp_gt)
        self.refexp_gt = refexp_gt
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.imgs.keys()
        self.predictions = {}
        self.k = k
        self.thresh_iou = thresh_iou

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def summarize(self):
        if dist.is_main_process():
            dataset2score = {
                "refcoco": {k: 0.0 for k in self.k},
                "refcoco+": {k: 0.0 for k in self.k},
                "refcocog": {k: 0.0 for k in self.k},
            }
            dataset2count = {"refcoco": 0.0, "refcoco+": 0.0, "refcocog": 0.0}
            mask_pos_num = 0
            mask_all_num = 0
            mask_ious_list = []
            for image_id in self.img_ids:
                ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
                assert len(ann_ids) == 1
                img_info = self.refexp_gt.loadImgs(image_id)[0]

                target = self.refexp_gt.loadAnns(ann_ids[0])
                prediction = self.predictions[image_id]
                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                target_bbox = target[0]["bbox"]
                converted_bbox = [
                    target_bbox[0],
                    target_bbox[1],
                    target_bbox[2] + target_bbox[0],
                    target_bbox[3] + target_bbox[1],
                ]
                giou = generalized_box_iou(sorted_boxes, torch.as_tensor(converted_bbox).view(-1, 4))
                for k in self.k:
                    if max(giou[:k]) >= self.thresh_iou:
                        dataset2score[img_info["dataset_name"]][k] += 1.0
                if 'masks' in prediction:
                    raw_scores = prediction["scores"]
                    sort_scores, sort_scores_order = raw_scores.sort(descending=True)
                    ordered_masks = prediction["masks"][sort_scores_order]
                    
                    rles = []
                    for mask in ordered_masks:
                        rle = mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]           
                        rle["counts"] = rle["counts"].decode("utf-8")
                        rles.append(rle)
                    
                    gt_rle = mask_util.frPyObjects(target[0]['segmentation'], img_info['height'], img_info['width'])[0]
                    gt_rle["counts"] = gt_rle["counts"].decode("utf-8")
                    
                    mask_ious = mask_util.iou(rles, [gt_rle], [False])
                    
                    mask_pos_num += int(mask_ious[0] > 0.7)
                    mask_all_num += 1
                    mask_ious_list.append(mask_ious[0])
                
                dataset2count[img_info["dataset_name"]] += 1.0

            for key, value in dataset2score.items():
                for k in self.k:
                    try:
                        value[k] /= dataset2count[key]
                    except:
                        pass
            results = {}
            for key, value in dataset2score.items():
                results[key] = sorted([v for k, v in value.items()])
                print(f" Dataset: {key} - Precision @ 1, 5, 10: {results[key]} \n")
            if 'masks' in prediction:
                mask_p = mask_pos_num / float(mask_all_num)
                mask_mean_iou = np.sum(mask_ious_list) / float(mask_all_num)
                print(f'Dataset: refcoco - Precision @ 1: {mask_p} - Mean IoU: {mask_mean_iou} \n')
            return results
        return None


def build(image_set, args):
    img_dir = Path(args.coco_path) / "train2014"

    refexp_dataset_name = args.refexp_dataset_name
    if refexp_dataset_name in ["refcoco", "refcoco+", "refcocog"]:
        if args.test:
            test_set = args.test_type
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_{test_set}.json"
        else:
            ann_file = Path(args.refexp_ann_path) / f"finetune_{refexp_dataset_name}_{image_set}.json"
    elif refexp_dataset_name in ["all"]:
        ann_file = Path(args.refexp_ann_path) / f"final_refexp_{image_set}.json"
    else:
        assert False, f"{refexp_dataset_name} not a valid datasset name for refexp"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = RefExpDetection(
        img_dir,
        ann_file,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=args.masks,
        return_tokens=True,
        tokenizer=tokenizer,
    )
    return dataset
