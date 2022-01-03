import streamlit as st
import os
import config
import tempfile
import numpy as np
import imutils
import time
import cv2


def detect_objects(test_vedio, confidence_threshold, nms_threshold):

	net = cv2.dnn.readNetFromDarknet(config.CONFIG_PATH, config.MODEL_PATH)
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

	video = cv2.VideoCapture(test_vedio)
	fps = video.get(cv2.CAP_PROP_FPS)
	writer_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	writer_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

	writer = None
	(W,H) = (None, None)

	try:
		prop = cv2.CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
		total = int(video.get(prop))

	except:
		total = -1

	while True:
		(grabbed, frame) = video.read()

		if not grabbed:
			break
		if W is None or H is None:
			(H,W) = frame.shape[:2]

		blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB = True, crop = False)
		start = time.time()
		net.setInput(blob)
		layerOutputs = net.forward(ln)
		end = time.time()


		boxes = []
		confidences = []
		classIDs = []

		for output in layerOutputs:
			for detection in output:
				score = detection[5:]
				classID = np.argmax(score)
				confidence = score[classID]

				if confidence > confidence_threshold:
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype('int')

					x = int(centerX - (width/2))
					y = int(centerY - (height/2))

					boxes.append([x,y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)


		idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
		if len(idxs) > 0:
			for i in idxs.flatten():
				(x,y) = (boxes[i][0], boxes[i][1])
				(w,h) = (boxes[i][2], boxes[i][3])

				color = [int(c) for c in config.COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
				text = f"{config.LABELS[classIDs[i]]}: {confidences[i]}"
				cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		if writer is None:
			print(writer)
			fourcc = cv2.VideoWriter_fourcc(*'H264')
			writer = cv2.VideoWriter(config.OUTPUT_PATH, fourcc, fps, (writer_width, writer_height), True)

			if total > 0:
				elap = (end - start)


		
		writer.write(frame)

	video.release()

	return total, elap



def main():

    st.title('Fire Detection')

    
    model = st.sidebar.selectbox('Model', ('Yolo-tinyv4', 'None'))
    if model == 'Yolo-tinyv4': 
        option = st.radio('', ['Choose video'])
        st.sidebar.title('Settings')
        confidence_slider = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, config.DEFALUT_CONFIDENCE, 0.05)
        print(option)
        if option == 'Choose video':
            test_videos = os.listdir(config.VIDEO_PATH)
            test_video = config.VIDEO_PATH + st.selectbox('Please choose video', test_videos)
        else:
            vedio = st.file_uploader('Upload a video', type = ['mp4'])
            if vedio is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(vedio.read())
                test_video = tfile.name
            else:
                st.write('** Please upload video **')


        if st.button ('Detect Fire'):
            
            st.write(f"Process Video..")
            total, elap = detect_objects(test_video, confidence_slider, config.NMS_THRESHOLD)
            output_video = open(config.OUTPUT_PATH, 'rb')
            video_bytes = output_video.read()

            st.write(f"Number of frames is  {total}")
            st.write(f"Times to process the  vedio {round((elap*total)/60, 2)} minutes")

            final_video = st.video(video_bytes)


if __name__ == '__main__':
    main()
    


