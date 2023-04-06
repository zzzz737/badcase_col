import os.path
import boto3
import pandas as pd
from boto3.session import Session
import yaml
import datetime
import math
import subprocess
import cv2
from skimage.measure import compare_ssim
import numpy as np
import onnxruntime
import shutil
from botocore.exceptions import ClientError
import tarfile
import json

def onehot2label(output):
    nms_output = []
    # cls_num = len(output[0]) - 5
    for result in output:
        clsid = np.argmax(result[5:])
        xc, yc, w, h = result[:4]
        score = result[4]
        nms_output.append([xc, yc, w, h, score, clsid])
    return np.array(nms_output)

def xywh2xyxy(input_w, input_h, origin_h, origin_w, x):
    """
    description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    param:
        origin_h:   height of original image
        origin_w:   width of original image
        x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
    return:
        y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
    """
    y = np.zeros_like(x)
    r_w = input_w / origin_w
    r_h = input_h / origin_h

    y[:, 0] = (x[:, 0] - x[:, 2] / 2) / r_w
    y[:, 2] = (x[:, 0] + x[:, 2] / 2) / r_w
    y[:, 1] = (x[:, 1] - x[:, 3] / 2) / r_h
    y[:, 3] = (x[:, 1] + x[:, 3] / 2) / r_h

    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    description: compute the IoU of two bounding boxes
    param:
        box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
        box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
        x1y1x2y2: select the coordinate format
    return:
        iou: computed iou
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                 np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def non_max_suppression(input_w, input_h, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
    """
    description: Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    param:
        prediction: detections, (x1, y1, x2, y2, conf, cls_id)
        origin_h: original image height
        origin_w: original image width
        conf_thres: a confidence threshold to filter detections
        nms_thres: a iou threshold to filter detections
    return:
        boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
    """
    # Get the boxes that score > CONF_THRESH
    boxes = prediction[prediction[:, 4] >= conf_thres]
    boxes[:, :4] = xywh2xyxy(input_w, input_h, origin_h, origin_w, boxes[:, :4])
    boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
    # Object confidence
    confs = boxes[:, 4]
    # Sort by the confs
    boxes = boxes[np.argsort(-confs)]
    # Perform non-maximum suppression
    keep_boxes = []
    while boxes.shape[0]:
        large_overlap = bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
        label_match = boxes[0, -1] == boxes[:, -1]
        # Indices of boxes with lower confidence scores, large IOUs and matching labels
        invalid = large_overlap & label_match
        keep_boxes += [boxes[0]]
        boxes = boxes[~invalid]
    boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
    return boxes

def post_process(input_w, input_h, output, origin_h, origin_w, CONF_THRESH, IOU_THRESHOLD):
    """
    description: postprocess the prediction
    param:
        output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
        origin_h:   height of original image
        origin_w:   width of original image
    return:
        result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
        result_scores: finally scores, a numpy, each element is the score correspoing to box
        result_classid: finally classid, a numpy, each element is the classid correspoing to box
    """
    # # Get the num of boxes detected
    pred = onehot2label(output[0])
    # Do nms
    boxes = non_max_suppression(input_w, input_h, pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
    result_boxes = boxes[:, :4] if len(boxes) else np.array([])
    result_scores = boxes[:, 4] if len(boxes) else np.array([])
    result_classid = boxes[:, 5] if len(boxes) else np.array([])
    return result_boxes, result_scores

class DownloadFromS3():
    def __init__(self, cfg):
        self.cfg = cfg
        self.access_key = self.cfg["download"]["S3"]["ACCESS_KEY"]
        self.secret_key = self.cfg["download"]["S3"]["SECRET_KEY"]
        self.url = self.cfg["download"]["S3"]["URL"]
        self.region_name = self.cfg["download"]["S3"]["REGION_NAME"]
        self.bucket_name = self.cfg["download"]["S3"]["DOWN_BUCKET_NAME"]

        # 连接s3
        self.session = Session(self.access_key, self.secret_key)
        self.s3_client = self.session.client('s3', region_name=self.region_name, endpoint_url=self.url, use_ssl=False,
                                             verify=False)

    def _get_db_key(self, orderID):
        prefix = "origin_file_pro"
        year = orderID[0: 2]
        month = orderID[2: 4]
        day = orderID[4: 6]
        db_key = prefix + '/' + year + '/' + month + '/' + day + '/' + orderID
        return db_key

    def _parase_xlsx(self, filepath):
        workbook = pd.read_excel(filepath, header=1)
        order_list = workbook["订单id"].tolist()
        device_list = workbook["设备id"].tolist()
        time_list = workbook["上报时间"].tolist()
        errtype_list = workbook["错误类型"].tolist()
        return order_list, device_list, time_list, errtype_list

    def check_file_exists_in_s3(self, s3_file_path):
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_file_path)
            return True
        except:
            return False

    def check_file_download(self, download_save_dir):
        if os.path.exists(download_save_dir):
            return True
        else:
            return False

    def _download_file(self, src_file_dir, deviceID, orderID):
        s3_video_path = src_file_dir + '/' + 'video.mp4'
        video_exist_flag = self.check_file_exists_in_s3(s3_video_path)
        s3_imgpost_path = src_file_dir + '/' + 'imgpost.txt'
        imgpost_exist_flag = self.check_file_exists_in_s3(s3_imgpost_path)
        s3_ai_result_path = src_file_dir + '/' + 'ai.tar.gz'
        ai_result_exist_flag = self.check_file_exists_in_s3(s3_imgpost_path)
        print(video_exist_flag, "------", imgpost_exist_flag)

        if video_exist_flag and imgpost_exist_flag and ai_result_exist_flag:
            dst_save_dir = os.path.join(self.cfg["save_path"], f'{deviceID}-{orderID}')
            if self.check_file_download(dst_save_dir):
                return True
            else:
                if not os.path.exists(dst_save_dir):
                    os.makedirs(dst_save_dir)
            dst_vedio_path = os.path.join(dst_save_dir, 'video.mp4')
            dst_imgpost_path = os.path.join(dst_save_dir, 'imgpost.txt')
            dst_ai_result_path = os.path.join(dst_save_dir, 'ai.tar.gz')
            self.s3_client.download_file(self.bucket_name, s3_video_path, dst_vedio_path)
            self.s3_client.download_file(self.bucket_name, s3_imgpost_path, dst_imgpost_path)
            self.s3_client.download_file(self.bucket_name, s3_ai_result_path, dst_ai_result_path)
            return True
        elif video_exist_flag == False:
            with open("error_order.txt", 'a+') as f:
                f.write(f"{s3_video_path} does not exists!" + '\n')
            return False
        elif imgpost_exist_flag == False:
            with open("error_order.txt", 'a+') as f:
                f.write(f"{s3_imgpost_path} does not exists!" + '\n')
            return False
        elif s3_ai_result_path == False:
            with open("error_order.txt", 'a+') as f:
                f.write(f"{s3_ai_result_path} does not exists!" + '\n')
            return False
        else:
            pass

        return True

    def main(self, xlsx_file_path):
        order_list, device_list, time_list, errtype_list = self._parase_xlsx(xlsx_file_path)
        for orderID, deviceID, timeID in zip(order_list, device_list, time_list):
            orderID = str(orderID)
            print(orderID)
            db_key = self._get_db_key(orderID)
            self._download_file(db_key, deviceID, orderID)

        return order_list, device_list, time_list, errtype_list

class GetBadcaseFrame():
    def __init__(self, cfg):
        self.cfg = cfg
        self.ffmpeg_exe = cfg["ffmpeg_path"]

        self.video_path = ""
        self.badcase_save_dir = ""
        self.bacase_frame_fps = cfg["download"]["badcase_frame_fps"]

    def _time_to_timestamp(self, time_str, format='%Y-%m-%d %H:%M:%S'):
        """将时间字符串转换为时间戳"""
        # 将时间字符串的毫秒提取出来，只留下年月日时分秒
        time = time_str.split('.')[0]
        millisecond = time_str.split('.')[1]
        # 将时间字符串解析为datetime对象
        time_obj = datetime.datetime.strptime(time, format)
        # 将datetime对象转换为时间戳
        # tmp = time_obj.timestamp()
        timestamp = float(time_obj.timestamp()) + float(millisecond)*0.001
        return timestamp

    def _get_badcase_frame_id(self, deviceID, orderID, time, errtype):
        order_dir = os.path.join(self.cfg["save_path"], f'{deviceID}-{orderID}')
        imgpost_path = os.path.join(order_dir, 'imgpost.txt')
        self.video_path = os.path.join(order_dir, 'video.mp4')
        self.badcase_save_dir = os.path.join(order_dir, 'badcase')
        if not os.path.exists(self.badcase_save_dir):
            os.makedirs(self.badcase_save_dir)

        timestamp = self._time_to_timestamp(time)

        with open(imgpost_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                imgID = int(line.strip().split(',')[0])
                gpstime = float(line.strip().split(',')[1])
                if errtype=="漏报":
                    enuvx = float(line.strip().split(',')[3])
                    enuvy = float(line.strip().split(',')[4])
                    enuvz = float(line.strip().split(',')[5])
                    speed = math.sqrt(enuvx * enuvx + enuvy * enuvy + enuvz * enuvz)
                    dtime = 8.0 / speed
                    if gpstime >= ((timestamp - dtime) - 0.001):
                        return deviceID, orderID, imgID
                elif errtype=="误报":
                    if gpstime >= timestamp:
                        return deviceID, orderID, imgID

    def _decode_ai_result(self, deviceID, orderID):
        tar_path = os.path.join(self.cfg["save_path"], f'{deviceID}-{orderID}', 'ai.tar.gz')
        decomp_path = os.path.join(self.cfg["save_path"], f'{deviceID}-{orderID}')
        if not os.path.exists(decomp_path):
            os.makedirs(decomp_path)
        try:
            t = tarfile.open(tar_path)
            t.extractall(path=decomp_path)
            t.close()
        except:
            return False
        return True

    def _decode_badcase_frame(self, deviceID, orderID, imgID):
        start_fram_id = imgID - self.bacase_frame_fps
        if start_fram_id < 0:
            start_fram_id = 0
        end_fram_id = imgID + self.bacase_frame_fps

        video_file = os.path.abspath(os.path.join(self.video_path))
        output_dir = os.path.abspath(os.path.join(self.badcase_save_dir))

        completed = subprocess.run(['bash', './decode.sh', video_file, output_dir, str(start_fram_id), str(end_fram_id), deviceID, orderID])
        if completed.returncode == 0:
            return True
        else:
            return False

class ImageDedup():
    def __init__(self, cfg):
        self.save_dir = cfg["save_path"]
        self.last_frame = None
        self.cal_ssim_img_size = (64, 64)
        self.ssim_thresh = 0.35

    def _cal_ssim(self, frame, last_frame):
        ssim = compare_ssim(frame, last_frame, multichannel=True)
        if ssim > self.ssim_thresh:
            return False
        else:
            return True

    def _GetSsim_f(self, frame):
        # print(type(frame))
        if type(frame) != np.ndarray:
            return False

        frame = cv2.resize(frame, self.cal_ssim_img_size, interpolation=cv2.INTER_CUBIC)
        if type(self.last_frame) != np.ndarray:
            self.last_frame = frame
            return False

        tag = self._cal_ssim(frame, self.last_frame)
        return tag

    def _process_image_dedup(self, deviceID, orderID):
        image_dedup_save_dir = os.path.join(self.save_dir, f'{deviceID}-{orderID}', 'imagededup')
        if not os.path.exists(image_dedup_save_dir):
            os.makedirs(image_dedup_save_dir)

        badcase_dir = os.path.join(self.save_dir, f'{deviceID}-{orderID}', 'badcase')
        imgfile_list = os.listdir(badcase_dir)
        for imgfile in imgfile_list:
            imgpath = os.path.join(badcase_dir, imgfile)
            frame = cv2.imread(imgpath)
            save_flag = self._GetSsim_f(frame)
            if save_flag:
                dst_img_path = os.path.join(image_dedup_save_dir, imgfile)
                shutil.copy(imgpath, dst_img_path)

        return True

class PreProcess():
    def __int__(self):
        pass
    def _resize(self, size, input_img):
        image = cv2.resize(input_img, tuple(size))
        return image
    def _normalize(self, mean, std, input_img):
        image = input_img.astype(np.float32) / 255.0
        image = (image - np.array([[mean]]) / np.array([[std]]))
        return image
    def _bgr2rgb(self, input_img):
        image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        return image
    def _transpose_channel(self, image):
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :, :, :]
        image = np.array(image, dtype=np.float32)
        return image
    def _preprocess_pipeline(self, preprocess_pipeline, input_img):
        # 依次执行处理步骤
        for step in preprocess_pipeline:
            if step["type"] == "resize":
                input_img = self._resize(step["size"], input_img)
            elif step["type"] == "normalize":
                input_img = self._normalize(step["mean"], step["std"], input_img)
            elif step["type"] == "bgr2rgb":
                input_img = self._bgr2rgb(input_img)
            elif step["type"] == "transpose":
                input_img = self._transpose_channel(input_img)
            else:
                raise ValueError("Invalid preprocess type: {}".format(step["type"]))
        return input_img

class OnnxInference():
    def __init__(self, cfg):
        self.cfg =cfg
        self.preprocess = PreProcess()

    def _preprocess(self, img, preprocess_pipeline):
        image = self.preprocess._preprocess_pipeline(preprocess_pipeline, img)
        return image

    def _inference(self, input_data, onnx_model):
        session = onnxruntime.InferenceSession(onnx_model)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name, session.get_outputs()[0].name], {input_name: input_data})
        return output[0]

    def _postprocess(self, output, src_img_wh, save_method, elem, edge=False, slicer_flag=False):
        src_img_bbox = []
        if save_method == "polygon":
            if output.ndim == 3:
                output = output[0].astype('uint8')
            elif output.ndim == 4:
                output = output[0][0].astype('uint8')
            if slicer_flag:
                up_pred_mask = np.zeros(output.shape)
                mask = np.vstack((up_pred_mask, output)).astype(np.uint8)
                mask = cv2.resize(mask, src_img_wh)
            else:
                mask = cv2.resize(output, src_img_wh)
            return mask, src_img_bbox
        elif save_method == "bbox":
            if edge == False:
                onnx_infer_size = self.cfg["Model"][elem]["onnx_size"]
                conf_thresh = self.cfg["Model"][elem]["conf_thresh"]
                iou_thresh = self.cfg["Model"][elem]["iou_thresh"]
                result_boxes, result_scores = post_process(onnx_infer_size[0], onnx_infer_size[1],
                                                           output, src_img_wh[1], src_img_wh[0], conf_thresh, iou_thresh)
                mask = np.zeros((src_img_wh[1], src_img_wh[0]), dtype=np.uint8)
                for bbox in result_boxes:
                    category_id = 1
                    xmin,ymin,xmax,ymax = bbox
                    polygon_box = np.array([[xmin,ymin], [xmin+(xmax-xmin), ymin], [xmax,ymax], [xmin, ymin+(ymax-ymin)]]).astype(np.int32)
                    cv2.fillPoly(mask, [polygon_box], category_id)
                return mask, result_boxes
        else:
            raise ValueError("Invalid model type: {}".format(type))

class GetBadcaseForLabeled():
    def __init__(self, cfg):
        self.cfg = cfg
        self.save_path = self.cfg["save_path"]
        self.onnx_inference = OnnxInference(self.cfg)

        self.badcase_save_dir = os.path.join(self.save_path, 'badcase')
        self._make_dirs(self.badcase_save_dir)
        self.badcase_vis_save_dir = os.path.join(self.save_path, 'badcse_vis')
        self._make_dirs(self.badcase_vis_save_dir)

    def _make_dirs(self, new_dir):
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    def _cal_miou(self, cloud_mask, edge_mask):
        """
        计算指定两张图的 mIOU
        """
        intersection = np.logical_and(cloud_mask, edge_mask)
        union = np.logical_or(cloud_mask, edge_mask)
        miou = np.sum(intersection) / np.sum(union)
        print(f"intersection: {np.sum(intersection)}, union: {np.sum(union)}")
        return miou

    def _save_badcase(self, src_img_path, deviceID, orderID, imgfile, elem):
        """
        将误检和漏检的图像和标注保存到指定目录
        """
        src_img_path = src_img_path
        elem_img_badcase_dir = os.path.join(self.cfg["save_path"], elem, 'badcase')
        self._make_dirs(elem_img_badcase_dir)
        elem_img_badcase_path = os.path.join(elem_img_badcase_dir, f'{imgfile}')
        shutil.copy(src_img_path, elem_img_badcase_path)

    def _xminyminwh2xyxy(self, bbox):
        xmin,ymin,w,h = bbox
        return [xmin, ymin, xmin+w, ymin+h]

    def _get_edge_mask_from_ai_result(self, imgfile, elem, img_w, img_h, save_method):
        deviceID = imgfile.split('-')[0]
        orderID = imgfile.split('-')[1]
        imgname = imgfile.split('-')[-1]
        src_img_box = []
        json_path = os.path.join(self.cfg["save_path"], f'{deviceID}-{orderID}', 'ai', f'{imgname[:-4]}.json')
        json_data = json.load(open(json_path, 'r'))
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if elem in json_data.keys():
            elem_objs = json_data[elem]
        else:
            return mask, src_img_box
        if save_method == 'polygon':
            for obj in elem_objs:
                category_id = 1
                polygon = obj["polygon"]
                contour = []
                for i in range(int(len(polygon)/2)):
                    x = polygon[i*2]
                    y = polygon[i*2 + 1]
                    contour.append([x,y])
                cv2.fillPoly(mask, [np.array(contour)], category_id)
            return mask, src_img_box
        elif save_method == 'bbox':
            for obj in elem_objs:
                bbox = obj["bbox"]
                xmin, ymin, w, h = bbox
                polygon_box = np.array([[xmin, ymin], [xmin+w, ymin], [xmin+w,ymin+h], [xmin,ymin+h]]).astype(np.int32)
                bbox = self._xminyminwh2xyxy(bbox)
                cv2.fillPoly(mask, [polygon_box], 1)

                src_img_box.append(bbox)
            return mask, src_img_box

    def _draw_polygon(self, src_img, mask):
        """
        将误检和漏检的图像和标注绘制出来
        """
        max_cls_id = np.max(mask)
        for cls_id in range(1, max_cls_id + 1):
            mask = np.where(mask == cls_id, 1, 0)
            contours, hierarchy = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            f_code = ''
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < 100: continue
                cv2.drawContours(src_img, [contour], -1, (0, 0, 255), 2)

        return src_img

    def _draw_bbox(self, src_img, bbox):
        for box in bbox:
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(src_img, (int(xmin),int(ymin)), (int(xmax), int(ymax)), [0, 255, 0], 2)

        return src_img

    def _get_mask(self, src_img, elem, imgfile, edge=False):
        src_img_wh = (src_img.shape[1], src_img.shape[0])
        save_method = self.cfg["objects"][elem]["save_method"]
        if edge:
            mask, src_img_box = self._get_edge_mask_from_ai_result(imgfile, elem, src_img.shape[1], src_img.shape[0], save_method)
            edge_box_vis = self._draw_bbox(src_img.copy(), src_img_box)
            cv2.imwrite('edge_box_vis.jpg', edge_box_vis)

            return mask, src_img_box
        else:
            preprocess_pipeline = self.cfg["Model"][elem]["preprocess"]
            model_path = self.cfg["Model"][elem]["model_path"]
            iuputdata = self.onnx_inference._preprocess(src_img, preprocess_pipeline)
            output = self.onnx_inference._inference(iuputdata, model_path)
            mask, src_img_bbox = self.onnx_inference._postprocess(output, src_img_wh, save_method, elem, edge)
            return mask, src_img_bbox

    def _get_signle_elem_mask(self, mask, save_method):
        """
        获得每一种要素的mask
        """
        if save_method == "polygon":   #polygon 中可能一个model中包含多个要素，需要将其剥离出来
            mask[mask > 0] = 1
            return mask
        elif save_method == "bbox":
            mask[mask > 0] = 1
            return mask
        else:
            pass


    def elements_process(self, src_img, src_img_path, deviceID, orderID, imgfile):
        # palette = [[0, 0, 0]]
        # for i in range(57):
        #     color = list(np.random.choice(range(256), size=3))
        #     palette.append(color)
        # palette = np.array(palette)
        """
        处理单张图像的预测结果和对应的标注信息，找出误检和漏检的情况
        """
        for elem in self.cfg["objects"]:
            if self.cfg["objects"][elem]["save_or_not"]:
                save_method = self.cfg["objects"][elem]["save_method"]
                # classid = self.cfg["objects"][elem]["edge_classid"]
                edge_mask, edge_bbox = self._get_mask(src_img, elem, imgfile, edge=True)
                cloud_mask, cloud_bbox = self._get_mask(src_img, elem, imgfile)

                if save_method == "bbox" and len(edge_bbox) == 0 and len(cloud_bbox) == 0:
                    miou = 1.0
                else:
                    # edge_mask_test = edge_mask.astype(int)
                    # pred = palette[edge_mask_test]
                    # visual_pic = cv2.addWeighted(src_img, 0.7, pred, 0.3, 0., dtype=cv2.CV_32F)
                    # cv2.imwrite('edge_visual.jpg', visual_pic)
                    #
                    # cloud_mask_test = cloud_mask.astype(int)
                    # pred = palette[cloud_mask_test]
                    # visual_pic = cv2.addWeighted(src_img, 0.7, pred, 0.3, 0., dtype=cv2.CV_32F)
                    # cv2.imwrite('cloud_visual.jpg', visual_pic)

                    # edge_mask = self._get_signle_elem_mask(edge_mask, classid, save_method, model_type="edge")
                    cloud_mask = self._get_signle_elem_mask(cloud_mask, save_method)

                    # print(f"single edge mask maxID: {np.max(edge_mask)}")
                    # print(f"single cloud mask maxID: {np.max(cloud_mask)}")
                    # edge_mask_test = edge_mask.astype(int)
                    # pred = palette[edge_mask_test]
                    # visual_pic = cv2.addWeighted(src_img, 0.7, pred, 0.3, 0., dtype=cv2.CV_32F)
                    # cv2.imwrite('edge_visual_single.jpg', visual_pic)

                    # cloud_mask_test = cloud_mask.astype(int)
                    # pred = palette[cloud_mask_test]
                    # visual_pic = cv2.addWeighted(src_img, 0.7, pred, 0.3, 0., dtype=cv2.CV_32F)
                    # cv2.imwrite('cloud_visual_single.jpg', visual_pic)

                    edge_badcase_vis_save_dir = os.path.join(self.cfg["save_path"], elem, 'edge')
                    self._make_dirs(edge_badcase_vis_save_dir)
                    edge_badcase_vis_save_path = os.path.join(edge_badcase_vis_save_dir, imgfile)

                    cloud_badcase_vis_save_dir = os.path.join(self.cfg["save_path"], elem, 'cloud')
                    self._make_dirs(cloud_badcase_vis_save_dir)
                    cloud_badcase_vis_save_path = os.path.join(cloud_badcase_vis_save_dir, imgfile)
                    if self.cfg["objects"][elem]["save_method"] == "polygon":
                        edge_mask_vis = self._draw_polygon(src_img.copy(), edge_mask)
                        cv2.imwrite(edge_badcase_vis_save_path, edge_mask_vis)

                        cloud_mask_vis = self._draw_polygon(src_img.copy(), cloud_mask)
                        cv2.imwrite(cloud_badcase_vis_save_path, cloud_mask_vis)
                    elif self.cfg["objects"][elem]["save_method"] == "bbox":
                        edge_box_vis = self._draw_bbox(src_img.copy(), edge_bbox)
                        cv2.imwrite(edge_badcase_vis_save_path, edge_box_vis)

                        cloud_box_vis = self._draw_bbox(src_img.copy(), cloud_bbox)
                        cv2.imwrite(cloud_badcase_vis_save_path, cloud_box_vis)
                    else:
                        pass

                    miou = self._cal_miou(cloud_mask, edge_mask)
                print(f"imgfile: {imgfile}, elem: {elem}, miou: {miou}")
                if miou < self.cfg["objects"][elem]["miou_thresh"]:
                    self._save_badcase(src_img_path, deviceID, orderID, imgfile, elem)

    def _process_order(self, deviceID, orderID):
        badcase_dir = os.path.join(self.save_path, f'{deviceID}-{orderID}', 'imagededup')
        imgfile_list = os.listdir(badcase_dir)
        imgfile_list.sort()
        for imgfile in imgfile_list:
            # print(imgfile)
            src_img_path = os.path.join(badcase_dir, imgfile)
            src_img = cv2.imread(src_img_path)
            self.elements_process(src_img, src_img_path, deviceID, orderID, imgfile)


def _init_class(cfg_path):
    with open(cfg_path, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

        down_from_s3_exe = DownloadFromS3(config)
        get_badcase_frame_exe = GetBadcaseFrame(config)
        image_dedup_exe = ImageDedup(config)
        # preprocess_exe = PreProcess()
        inference_exe = OnnxInference(config)
        save_badcase_exe = GetBadcaseForLabeled(config)

    return down_from_s3_exe, get_badcase_frame_exe, image_dedup_exe, inference_exe, save_badcase_exe

def batch_test():
    cfg_path = "./config.yaml"
    down_from_s3_exe, get_badcase_frame_exe, image_dedup_exe, inference_exe, save_badcase_exe = _init_class(cfg_path)

    xlsx_dir = "./xlsx_dir"
    xlsx_lists = os.listdir(xlsx_dir)
    for xlsx_file in xlsx_lists:
        xlsx_file_path = os.path.join(xlsx_dir, xlsx_file)
        order_list, device_list, time_list, errtype_list = down_from_s3_exe._parase_xlsx(xlsx_file_path)
        for orderID, deviceID, time, errtype in zip(order_list, device_list, time_list, errtype_list):
            orderID = str(orderID)
            db_key = down_from_s3_exe._get_db_key(orderID)
            down_flag = down_from_s3_exe._download_file(db_key, deviceID, orderID)
            if down_flag:  # 判断文件是否下载成功
                deviceID, orderID, imgID = get_badcase_frame_exe._get_badcase_frame_id(deviceID, orderID, time, errtype)
                decode_flag = get_badcase_frame_exe._decode_badcase_frame(deviceID, orderID, imgID)
                get_badcase_frame_exe._decode_ai_result(deviceID, orderID)
                if decode_flag:  # 判断文件是否解压成功
                    dedup_flag = image_dedup_exe._process_image_dedup(deviceID, orderID)
                    if dedup_flag:  # 判断文件是否去重成功
                        save_badcase_exe._process_order(deviceID, orderID)
                    else:
                        print(f"Dedup {orderID} failed!")
                else:
                    print(f"Decode {orderID} failed!")
            else:
                print(f"Download {orderID} failed!")


if __name__ == "__main__":
    # test_inference_result()
    batch_test()