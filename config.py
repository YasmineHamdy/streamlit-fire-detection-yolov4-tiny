import numpy as np

# Docker Path
DIR_PATH = ''

model = 'models/fire-dataset-weight/yolov4-tiny_best.weights'
model_config = 'models/yolov4-tiny.cfg'
labels = 'models/classes.names'
input_videos = 'test-data/vedios/'
output_video = 'output/output_video.mp4'

MODEL_PATH = DIR_PATH + model
CONFIG_PATH = DIR_PATH + model_config
LABEL_PATH = DIR_PATH + labels
OUTPUT_PATH = DIR_PATH + output_video
INPUT_PATH = DIR_PATH+ input_videos
VIDEO_PATH = DIR_PATH + input_videos

LABELS = open(LABEL_PATH).read().strip().split('\n')

COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype = 'uint8')

DEFALUT_CONFIDENCE = 0.5
NMS_THRESHOLD = 0.3

