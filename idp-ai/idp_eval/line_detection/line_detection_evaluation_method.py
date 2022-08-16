from enum import Enum

import numpy as np
from pathlib import Path

import json

from mean_average_precision import MeanAveragePrecision2d

from idp_eval.line_detection.dataset_tools import calculate_TP
import csv
IOU_thresh = 0.1
line_half_width = 5


class Metric:
    def __init__(self):
        self.metric_fn = MeanAveragePrecision2d(num_classes=1)
        self.tp = 0
        self.total = 0
        self.mode = ""
    def create_metric_fn(self, gt_json_path, dt_json_path):
        gt_data = json.load(open(gt_json_path))
        dt_data = json.load(open(dt_json_path))
        for num in range(len(gt_data)):
            gt = [list(map(int, x)) for x in gt_data[str(num + 1)]]
            dt = [list(map(int, x[1])) for x in dt_data[str(num + 1)]]
            gt = [[0, 0, 0, 0]] if not gt else gt
            dt = [[0, 0, 0, 0]] if not dt else dt
            gt_rc = np.array(gt) + np.array([-1, -1, 1, 1]) * line_half_width
            dt_rc = np.array(dt) + np.array([-1, -1, 1, 1]) * line_half_width  # width = 10px

            gt_ap = [x + [0, 0, 0] for x in gt]
            gt_ap = np.array(gt_ap) + np.array([-1, -1, 1, 1, 0, 0, 0]) * line_half_width

            dt_ap = [x + [0, 1.0] for x in dt]
            # dt_ap = [x for x in dt_ap if (x[0]==x[2] or x[1]==x[3])]
            dt_ap = np.array(dt_ap) + np.array([-1, -1, 1, 1, 0, 0]) * line_half_width  # width = 10px

            self.metric_fn.add(dt_ap, gt_ap)
            self.tp += calculate_TP(dt_rc, gt_rc, IOU_thresh)
            self.total += len(gt)


    def evaluate_line_detection(self, ground_truth_path, detection_result_path):

        if Path(ground_truth_path).is_dir() and Path(detection_result_path).is_dir():
            self.mode = "folder"
            ground_truth_json_list = list(ground_truth_path.glob("*/*.json"))

            for gt_json_path in ground_truth_json_list:
                dt_json_path = detection_result_path / (
                        gt_json_path.parent.name + "/" + gt_json_path.stem[:-4] + "lines.json")
                self.create_metric_fn(gt_json_path, dt_json_path)
        elif Path(ground_truth_path).is_file() and Path(detection_result_path).is_file():
            self.mode = "file"
            self.create_metric_fn(ground_truth_path, detection_result_path)
        else:
            print("The Paths provided are not valid")
            return

    def print_metric(self):
        print(f"\nEvaluate the detection result by",self.mode)
        print(f'Recall at IOU={IOU_thresh}:', self.tp / self.total)
        # compute PASCAL VOC metric
        print(f"VOC PASCAL mAP: {self.metric_fn.value(iou_thresholds=[0.1], recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")
        # compute PASCAL VOC metric at the all points
        print(f"VOC PASCAL mAP in all points: {self.metric_fn.value(iou_thresholds=[0.1])['mAP']}")
        # compute metric COCO metric
        print(
            f"COCO mAP: {self.metric_fn.value(iou_thresholds=list(np.arange(0.1, 1.0, 0.1)), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")



    # json_list = list(ground_truth_path.glob("*/*.json"))
    # a = 1
    # metric = Metric()
    # for json_path in json_list:
    #     ground_truth_path = json_path
    #     detection = detection_result_path / (json_path.parent.name+"/"+ json_path.stem[:-4]+"lines.json")
    #     print(detection.name)
    #     metric.evaluate_line_detection(ground_truth_path, detection)
    #
    #     with open('eggs.csv', 'a', newline='') as csvfile:
    #         fieldnames = ['filename', 'Recall at IOU', 'VOC PASCAL mAP', 'VOC PASCAL mAP in all points', 'COCO mAP']
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #         if a==1:
    #             writer.writeheader()
    #             a=2
    #         writer.writerow({'filename':detection.name,
    #                             'Recall at IOU':str(metric.tp / metric.total),
    #                             'VOC PASCAL mAP':str(metric.metric_fn.value(iou_thresholds=[0.1], recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']),
    #                            'VOC PASCAL mAP in all points':str(metric.metric_fn.value(iou_thresholds=[0.1])['mAP']),
    #                            'COCO mAP':str(metric.metric_fn.value(iou_thresholds=list(np.arange(0.1, 1.0, 0.1)), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP'])})