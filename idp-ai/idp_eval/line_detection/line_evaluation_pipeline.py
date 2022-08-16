import io
import os
import fitz
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

import json
import shutil

from mean_average_precision import MeanAveragePrecision2d

from idp_doc.service.OCR_operation import OCR_transform_and_get_word, pdf2OCR
from idp_doc.service.line_detection import image_to_lines_info
from idp_eval.line_detection.dataset_tools import calculate_TP

IOU_thresh = 0.1
line_half_width = 5
draw_width = 2
close_thresh = 10


class Evaluate_line_detection_pipeline():
    def __init__(self, dataset_pdf, dataset_json, get_ground_truth, detection, first_page=True, draw=True,
                 calculate=True, print=False, extract=False):
        self.dataset_pdf = dataset_pdf
        self.dataset_json = dataset_json
        self.get_ground_truth = get_ground_truth
        self.detection = detection
        self.first_page = first_page
        self.draw = draw
        self.draw_anno = False
        self.calculate = calculate
        self.print = print
        self.extract = extract
        self.metric_fn = MeanAveragePrecision2d(num_classes=1)
        self.tp = 0
        self.total = 0

    def evaluate_line_detection_pipeline(self):
        self.original_pdf_list = list(self.dataset_pdf.glob('*/*original.pdf'))
        if self.get_ground_truth:
            # generate ground_truth from pdf with annotation
            self.anno_pdf_list = list(self.dataset_pdf.glob('*/*anno.pdf'))
            self.generate_ground_truth()
            self.anno_json_list = list(self.dataset_pdf.glob('*/*anno.json'))
            self.process_ground_truth()
        if self.detection:
            # run line detection process and get detection result
            self.get_detected_lines()
        if self.draw:
            # draw detection
            self.draw_detection_result()
        if self.calculate:
            # calculate accuracy
            self.create_metric_fn()
        if self.print:
            self.print_metric()
        if self.extract:
            self.extract_json_from_dataset_for_evaluation()

    def generate_ground_truth(self):
        for anno_pdf in self.anno_pdf_list:

            doc = fitz.open(str(anno_pdf))
            ground_truth_dict = {}
            page_ground_truth_ls = []
            # get dataset pdf's width by the corresponding json
            json_path = anno_pdf.with_name(anno_pdf.stem[:-4] + 'OCR.json')
            OCR_result = json.load(open(json_path, 'r'))
            width, height = OCR_result['files'][0]['pages'][0]['bbox'][2:4]
            for num, page in enumerate(doc, 1):
                # get scale
                pix = page.getPixmap(matrix=fitz.Matrix(1, 1))
                scale = width / pix.width
                # save anno rect to json one by one, page by page
                annot = page.annots()
                for anno in annot:
                    x1, y1, x2, y2 = anno.rect
                    if anno.type[1] == 'Line':
                        if y2 - y1 <= x2 - x1:  # horizontal
                            y = (y1 + y2) / 2
                            l = list(map(int, [i * scale for i in [x1, y, x2, y]]))
                        else:  # y2 - y1> x2 - x1 # vertical
                            x = (x1 + x2) / 2
                            l = list(map(int, [i * scale for i in [x, y1, x, y2]]))

                        if not (l[0] == l[2] and l[1] == l[3]):
                            page_ground_truth_ls.append(l)
                    elif anno.type[1] == 'Square':
                        square_anno = np.array([[x1, y1, x2, y1], [x1, y1, x1, y2], [x2, y1, x2, y2], [x1, y2, x2, y2]])
                        square_anno = (square_anno * scale).astype(int).tolist()
                        page_ground_truth_ls.extend(square_anno)
                ground_truth_dict[num] = page_ground_truth_ls

                json_path = anno_pdf.parent / (anno_pdf.stem + '.json')
                with open(json_path, 'w') as f:
                    json.dump(ground_truth_dict, f)

    def process_ground_truth(self):
        for anno_json in self.anno_json_list:
            anno_json_data = json.load(open(str(anno_json), 'r'))
            for page_num, lines in anno_json_data.items():
                hori_lines, verti_lines = [], []
                for line in lines:
                    if line[2] - line[0] >= line[3] - line[1]:
                        hori_lines.append(line)
                    else:
                        verti_lines.append(line)

                hori_lines.sort(key=lambda x: x[1])
                current_index = 0
                for index, line in enumerate(hori_lines):
                    if index != 0:
                        current_line = hori_lines[current_index]
                        if line[1] - current_line[1] < close_thresh:
                            if line[0] <= current_line[2] and line[2] >= current_line[0]:
                                y = int((line[1] + current_line[1]) / 2)
                                hori_lines[current_index] = [min(line[0], current_line[0]), y,
                                                             max(line[2], current_line[2]), y]
                                hori_lines[index] = [0, 0, 0, 0]
                                continue
                    current_index = index

                verti_lines.sort(key=lambda x: x[0])
                current_index = 0
                for index, line in enumerate(verti_lines):
                    if index != 0:
                        current_line = verti_lines[current_index]
                        if line[0] - current_line[0] < close_thresh:
                            if line[1] <= current_line[3] and line[3] >= current_line[1]:
                                x = int((line[0] + current_line[0]) / 2)
                                verti_lines[current_index] = [x, min(line[1], current_line[1]), x,
                                                              max(line[3], current_line[3])]
                                verti_lines[index] = [0, 0, 0, 0]
                                continue
                    current_index = index

                anno_json_data[page_num] = [i for i in hori_lines + verti_lines if i != [0, 0, 0, 0]]
            json.dump(anno_json_data, open(str(anno_json), 'w'))

    def detect_lines_by_page(self, pdf_data, ocr_data_transs):
        doc = fitz.Document(stream=pdf_data, filetype='PDF')
        lines_rects = {}
        for num, page in enumerate(doc, 0):
            pix = page.getPixmap(matrix=fitz.Matrix(1, 1))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            lines_rect = image_to_lines_info(img, ocr_data_transs[num]['pages'][0]['paragraphs'])
            lines_rects[num + 1] = lines_rect
            if self.first_page:
                break
        return lines_rects

    def get_detected_lines(self):
        lang = 'EN'
        for pdf_path in self.original_pdf_list:
            doc = fitz.open(str(pdf_path))
            buff = io.BytesIO()
            doc.save(buff)
            pdf_data = buff.getvalue()

            filename = pdf_path.name
            ress, images = pdf2OCR(pdf_data, lang)
            ocr_data_transs = OCR_transform_and_get_word(filename, ress, images, [0] * len(images), lang)
            lines_rects = self.detect_lines_by_page(pdf_data, ocr_data_transs)
            json_path = pdf_path.parent / (pdf_path.stem[:-8] + 'lines.json')
            with open(json_path, 'w') as f:
                json.dump(lines_rects, f)

    def draw_detection_result(self):
        for original_pdf in self.original_pdf_list:
            fp = open(str(original_pdf), 'rb')
            bin_in = io.BytesIO(fp.read())
            doc = fitz.Document(stream=bin_in.read(), filetype='PDF')
            im_list = []
            for num, page in enumerate(doc, 1):
                M = fitz.Matrix(1, 1)
                pix = page.getPixmap(matrix=M)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img = np.ascontiguousarray(img[..., [2, 1, 0]])
                if self.draw_anno:
                    d = json.load(open(str(original_pdf.with_name(original_pdf.stem[:-8] + 'anno.json'))))
                    if d.get(str(num)) and len(d.get(str(num))) > 0:
                        gt = [list(map(int, x)) for x in d[str(num)]]
                        for l in gt:
                            x1, y1, x2, y2 = l
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), draw_width)

                d = json.load(open(str(original_pdf.with_name(original_pdf.stem[:-8] + 'lines.json'))))
                if d.get(str(num)) and len(d.get(str(num))) > 0:
                    dt = [list(map(int, x[1])) for x in d[str(num)]]
                    for l in dt:
                        x1, y1, x2, y2 = l
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), draw_width)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(img, 'RGB')
                im_list.append(im)
                if self.first_page:
                    break
            pdf_name = original_pdf.with_name(original_pdf.stem[:-8] + 'lines.pdf')
            im_list[0].save(str(pdf_name), 'PDF', save_all=True, append_images=im_list[1:])

    def create_metric_fn(self):
        for anno_json in self.anno_json_list:
            gt_data = json.load(open(anno_json))
            dt_data = json.load(open(anno_json.with_name(anno_json.stem[:-4] + 'lines.json')))
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
                if self.first_page:
                    break
                self.metric_fn.add(dt_ap, gt_ap)
                self.tp += calculate_TP(dt_rc, gt_rc, IOU_thresh)
                self.total += len(gt)

    def print_metric(self):
        print(f'Recall at IOU={IOU_thresh}:', self.tp / self.total)
        # compute PASCAL VOC metric
        print(
            f"VOC PASCAL mAP: {self.metric_fn.value(iou_thresholds=[0.1], recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")
        # compute PASCAL VOC metric at the all points
        print(f"VOC PASCAL mAP in all points: {self.metric_fn.value(iou_thresholds=[0.1])['mAP']}")
        # compute metric COCO metric
        print(
            f"COCO mAP: {self.metric_fn.value(iou_thresholds=list(np.arange(0.1, 1.0, 0.1)), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")

    def extract_json_from_dataset_for_evaluation(self):
        ground_truth_folder = self.dataset_json / "line_detection_ground_truth"
        result_folder = self.dataset_json / "line_detection_result"
        if not ground_truth_folder.is_dir(): ground_truth_folder.mkdir()
        if not result_folder.is_dir(): result_folder.mkdir()
        anno_file_list = list(self.dataset_pdf.glob("*/*anno.json"))
        lines_file_list = list(self.dataset_pdf.glob("*/*lines.json"))

        for json_path in anno_file_list:
            new_json_path = ground_truth_folder / (json_path.parent.name + "/" + json_path.name)
            if not new_json_path.parent.is_dir(): new_json_path.parent.mkdir()
            shutil.copy(str(json_path), new_json_path)
        for json_path in lines_file_list:
            new_json_path = result_folder / (json_path.parent.name + "/" + json_path.name)
            if not new_json_path.parent.is_dir(): new_json_path.parent.mkdir()
            shutil.copy(str(json_path), new_json_path)
