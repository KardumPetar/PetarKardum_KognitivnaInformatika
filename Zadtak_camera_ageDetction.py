import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
from deepface import DeepFace 

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)


face1 = "Modeli/opencv_face_detector.pbtxt"
face2 = "Modeli/opencv_face_detector_uint8.pb"
age1 = "Modeli/age_deploy.prototxt"
age2 = "Modeli/age_net.caffemodel"
gen1 = "Modeli/gender_deploy.prototxt"
gen2 = "Modeli/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


la = ['0-2', '4-6', '8-12', '15-20',
      '25-32', '38-43', '48-53', '60-100']
lg = ['M', 'Ž']


def detect_face(vid):
    face = cv2.dnn.readNet(face2, face1)
    age = cv2.dnn.readNet(age2, age1)
    gen = cv2.dnn.readNet(gen2, gen1)
    vid_copy = vid.copy()
    vid_copy = cv2.resize(vid_copy, (720, 640))

    fr_h = vid_copy.shape[0]
    fr_w = vid_copy.shape[1]
    blob = cv2.dnn.blobFromImage(vid_copy, 1.0, (300, 300),[104, 117, 123], True, False)

    face.setInput(blob)
    detections = face.forward()
    
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:

            x1 = int(detections[0, 0, i, 3]*fr_w)
            y1 = int(detections[0, 0, i, 4]*fr_h)
            x2 = int(detections[0, 0, i, 5]*fr_w)
            y2 = int(detections[0, 0, i, 6]*fr_h)
            
            faceBoxes.append([x1, y1, x2, y2])
            
            cv2.rectangle(vid_copy, (x1, y1), (x2, y2),(0, 255, 0), int(round(fr_h/150)), 8) 

    
    if not faceBoxes:
        #print("Nije detektirano lice.")
        cv2.putText(vid_copy,
                    f"Nije detektirano lice.",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (217, 0, 0),
                    2,
                    cv2.LINE_AA)

   
    for faceBox in faceBoxes:
        
        face = vid_copy[max(0, faceBox[1]-15):min(faceBox[3]+15, vid_copy.shape[0]-1),max(0, faceBox[0]-15):min(faceBox[2]+15,vid_copy.shape[1]-1)]
        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        gen.setInput(blob)
        genderPreds = gen.forward()
        gender = lg[genderPreds[0].argmax()]

        age.setInput(blob)
        agePreds = age.forward()
        age_int = la[agePreds[0].argmax()]

        emotion=""
        result = DeepFace.analyze(face,actions=['emotion'], enforce_detection=False)
        emotion = result[0]["dominant_emotion"] 

        cv2.putText(vid_copy,
                    f'{gender}, {age_int}, {emotion}',
                    (faceBox[0]-150, faceBox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (217, 0, 0),
                    2,
                    cv2.LINE_AA)
        
    return vid_copy


holdColor="f"
while True:
    result, video_frame = video_capture.read() 
    
    if result is False:
        break 
    
    video_frame = detect_face(video_frame)
    
    r=video_frame.copy()
    r[:, :, 0] = 0
    r[:, :, 1] = 0
    b=video_frame.copy()
    b[:, :, 1] = 0
    b[:, :, 2] = 0
    g=video_frame.copy()
    g[:, :, 2] = 0
    g[:, :, 0] = 0 


    if cv2.waitKey(1) & 0xFF == ord("f"):
        holdColor="f"
    if cv2.waitKey(1) & 0xFF == ord("r"):
        holdColor="r"
    if cv2.waitKey(1) & 0xFF == ord("b"):
        holdColor="b"
    if cv2.waitKey(1) & 0xFF == ord("g"):
        holdColor="g"

    img=video_frame
    if holdColor=="r":
        img=r
    if holdColor=="b":
        img=b
    if holdColor=="g":
        img=g

    
    cv2.imshow("My Face Detection Project", img) 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()