import cv2
import os
import numpy as np

# =============================
# Konfigurasi
# =============================
# Nama orang
person_name = "Justin"

# Output folder
output_dir = rf"C:/Users/justf/Desktop/Sem 6/KP/FaceRecog/Dataset_Face/{person_name}"
os.makedirs(output_dir, exist_ok=True)

# Path YOLOv4 Tiny Face
cfg_path = "C:/Users/justf/Desktop/Sem 6/KP/FaceRecog/yolov4.cfg"
weights_path = "C:/Users/justf/Desktop/Sem 6/KP/FaceRecog/yolov4.weights"

# Load YOLO
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# =============================
# Pilih sumber video
# =============================
# Webcam
cap = cv2.VideoCapture(0)

# Kalau mau pakai video file:
# cap = cv2.VideoCapture("path_to_video.mp4")

print("✅ Mulai pengambilan dataset.")
print("Tekan 'q' untuk berhenti.")

saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    ln = net.getUnconnectedOutLayersNames()
    outputs = net.forward(ln)

    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            confidence = scores[0]
            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                (cX, cY, width, height) = box.astype("int")
                x = int(cX - width/2)
                y = int(cY - height/2)
                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Draw & save
    if len(idxs) > 0:
        for i in idxs.flatten():
            x,y = boxes[i][0], boxes[i][1]
            w_box,h_box = boxes[i][2], boxes[i][3]
            cv2.rectangle(frame, (x,y), (x+w_box,y+h_box), (0,255,0),2)

            face_crop = frame[y:y+h_box, x:x+w_box]
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (160,160))
                file_path = os.path.join(output_dir, f"{person_name}_{saved_count:03d}.jpg")
                cv2.imwrite(file_path, face_resized)
                saved_count +=1

    cv2.putText(frame, f"Saved: {saved_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Collect Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"/n✅ Selesai. Total gambar disimpan: {saved_count}")
