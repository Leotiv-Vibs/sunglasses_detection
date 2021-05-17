import cv2
import numpy as np
import glob
import random

# Важно указать правильный путь к файлу
# Load Yolo
modelWeights = "/home/leotiv/PycharmProjects/object_detection/yolov3_training_last.weights"

net = cv2.dnn.readNet(modelWeights, "yolov3_testing.cfg")

# Name custom object
classes = ["sunglasses"]

# Важно указать правильный путь к изображениям
# Images path
images_path = glob.glob(r"/home/leotiv/Изображения/images/images/images/*.jpg")
print(images_path)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

random.shuffle(images_path)
for img_path in images_path:

    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=1.3, fy=1.3)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])

            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

    cv2.imshow("Image", img)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()
