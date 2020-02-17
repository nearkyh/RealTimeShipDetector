'''
Example usage:

    python detector.py \
        --video=file_path \
        --model=model_name

'''

import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
import argparse
import time

# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


parser = argparse.ArgumentParser()
parser.add_argument('--video', default='test.mp4', type=str, help="Video to test.")
parser.add_argument('--camera', default=None, type=int, help="Device index of the camera.")
parser.add_argument('--model', default='ssd_inception_ship_v3', type=str, help='Saved model name.')
args = parser.parse_args()


class ShipDetection:

    def __init__(self, model):
        # Model preparation
        self.model = 'object_detection/saved_models/{}'.format(model)
        self.path_to_ckpt = self.model + '/frozen_inference_graph.pb'
        self.path_to_labels = os.path.join('object_detection/data', 'ship_label_map.pbtxt')
        self.num_classes = 7

        # Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Loading label map
        self.label_map = label_map_util.load_labelmap(self.path_to_labels)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,
                                                                         max_num_classes=self.num_classes,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        # Init visualization
        self.min_score_thresh = .5
        self.line_thickness = 4

    def run(self, image, display=True):
        image_np_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded}
        )
        # Whether visualization
        if display==True:
            self.visualization(image, boxes, scores, classes)
        elif display==False:
            pass

        return boxes, scores, classes, num_detections

    def visualization(self, image, boxes, scores, classes):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            min_score_thresh=self.min_score_thresh,
            line_thickness=self.line_thickness)

    def data_process(self, boxes, scores, classes, num_detections):
        get_scores = np.squeeze(scores)
        get_category = np.array([self.category_index.get(i) for i in classes[0]])
        get_boxes = np.squeeze(boxes)

        count_objects = 0
        count_category = np.array([])
        count_score = np.array([])
        for i in range(len(get_scores)):
            if scores is None or get_scores[i] > self.min_score_thresh:
                count_objects = count_objects + 1
                count_category = np.append(count_category, get_category[i])
                count_score = np.append(count_score, get_scores[i])
        '''
        (x1,y1) --------
            |          |
            |          |
            |          |
            ------ (x2,y2)
        '''
        height, width, _ = image_np.shape
        count_point = np.array([])
        for i in range(len(count_score)):
            # Get boxes index : [y1, x1, y2, x2]
            box_point = get_boxes[i]
            x1, y1 = (box_point[1] * width), (box_point[0] * height)
            x2, y2 = (box_point[3] * width), (box_point[2] * height)
            point_x = (x1 + y1) / 2
            point_y = (x2 + y2) / 2
            point = (point_x, point_y)
            count_point = np.append(count_point, point)

        return count_objects, count_category, count_score, count_point


class VideoRecorder:

    def __init__(self):
        pass

    def set_record(self, savePath='.', fileName='test', width=640, height=480, frameRate=30.0):
        recording_video = "{}/rec_{}.avi".format(savePath, fileName)
        fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

        return cv2.VideoWriter(recording_video, fcc, frameRate, (width, height))

    def get_record(self, frame, set_record):
        set_record.write(frame)


if __name__ == '__main__':

    video_fname = args.video

    # ================================= #
    #       Define ship detection       #
    # ================================= #
    shipDetection = ShipDetection(args.model)

    # ================================ #
    #       Define video capture       #
    # ================================ #
    if args.camera == None:
        cap = cv2.VideoCapture(video_fname)
    else:
        cap = cv2.VideoCapture(args.camera)
        # Set video resolution (1280, 720)
        cap.set(3, 1280)
        cap.set(4, 720)

    # ================================= #
    #       Define video recorder       #
    # ================================= #
    videoRecorder = VideoRecorder()
    rec_fname = video_fname.split('/')[-1]
    savePath = './test_videos/result'
    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    else:
        pass
    set_record = videoRecorder.set_record(
                    savePath=savePath,
                    fileName="{}_{}".format(rec_fname[:-4], args.model),
                    width=int(cap.get(3)),
                    height=int(cap.get(4)),
                    frameRate=30.0
    )

    # ======================= #
    #       Define init       #
    # ======================= #
    count_frame = 0
    prevTime = 0
    processing_speed = []

    # ========================== #
    #       For evaluation       #
    # ========================== #
    list_recall = []
    count_detect = 0
    count_miss = 0
    list_precision = []

    while True:
        try:
            ret, image_np = cap.read()

            # ==================================== #
            #       Resize Image (1280, 720)       #
            # ==================================== #
            try:    
                if image_np.shape != (720, 1280, 3):
                    image_np = cv2.resize(image_np, (1280, 720))
            except Exception as e:
                pass

            # ================================ #
            #       Run object detection       #
            # ================================ #
            boxes, scores, classes, num_detections = shipDetection.run(image=image_np)
            count_objects, count_category, count_score, count_point = shipDetection.data_process(
                                                                            boxes=boxes,
                                                                            scores=scores,
                                                                            classes=classes,
                                                                            num_detections=num_detections
            )
            # print(count_objects, count_category, count_score, count_point)

            try:
                # ================================= #
                #       AP(Average Precision)       #
                # ================================= #
                for i in count_score:
                    list_precision.append(float(i))
                ap = sum(list_precision) / len(list_precision)
                ap = round(ap, 4)
                # print("AP:", ap)

                # ================== #
                #       Recall       #
                # ================== #
                for i in count_category:
                    if i['name'] == rec_fname[:-4]:
                        list_recall.append('T')
                        count_detect += 1
                    if i['name'] != rec_fname[:-4]:
                        list_recall.append('F')
                        count_miss += 1
                recall = count_detect / (count_detect + count_miss)
                recall = round(recall, 4)
                # print("Recall:", recall)
            except Exception as e:
                # print(e)
                pass

            # ====================== #
            #       Frame Rate       #
            # ====================== #
            curTime = time.time()
            sec = curTime - prevTime
            prevTime = curTime
            frameRate = "%0.1f" % (1 / (sec))

            # ============================ #
            #       Processing Speed       #
            # ============================ #
            speed = round(sec * 1000.0, 1)
            avg_speed = 0
            try:
                processing_speed.append(sec)
                avg_speed = (sum(processing_speed[1:]) / len(processing_speed[1:])) * 1000.0
                avg_speed = round(avg_speed, 1)
                # print("Speed:", avg_speed)
            except Exception as e:
                print(e)

            # =================== #
            #       Display       #
            # =================== #
            # cv2.rectangle(image_np, (3, 3), (50+int(len(args.model))*10, 50), (0, 0, 0), -1)
            # cv2.rectangle(image_np, (3, 3), (45+int(len(args.model))*2, 25), (0, 0, 0), -1)
            # cv2.putText(image_np, "Model: {}".format(args.model), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            # cv2.putText(image_np, "FPS: {}".format(frameRate), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            # cv2.putText(image_np, "AP: {}".format(ap), (5, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            # cv2.putText(image_np, "Recall: {}".format(recall), (5, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.imshow('Ship Detector', cv2.resize(image_np, (1280,720)))

            # ===================== #
            #       Save data       #
            # ===================== #
            try:
                if count_objects > 0:
                    count_frame += 1
                    data_value = [(count_frame, count_category[0]['name'], count_score[0], float(speed), float(frameRate))]
                else:
                    count_frame += 1
                    data_value = [(count_frame, None, None, None, None)]
                column_name = ['frame', 'category', 'score', 'speed(ms)', 'fps']
                time_series_data = 'time_series_data/'
                category = (video_fname.split('/')[-1])
                category = "{}_{}".format(category[:-4], args.model)
                if not os.path.isdir(time_series_data):
                    os.mkdir(time_series_data)
                save_fname = time_series_data + '{}.csv'.format(category)
                if os.path.exists(save_fname) == True:
                    with open(save_fname, 'a') as f:
                        xml_df = pd.DataFrame(data_value, columns=column_name)
                        xml_df.to_csv(f, header=False, index=None)
                else:
                    xml_df = pd.DataFrame(columns=column_name)
                    xml_df.to_csv(save_fname, index=None)
                    with open(save_fname, 'a') as f:
                        xml_df = pd.DataFrame(data_value, columns=column_name)
                        xml_df.to_csv(f, header=False, index=None)
            except Exception as e:
                print(e)

            # =========================== #
            #       Recording Video       #
            # =========================== #
            videoRecorder.get_record(frame=image_np, set_record=set_record)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(e)

    cap.release()
    cv2.destroyAllWindows()
