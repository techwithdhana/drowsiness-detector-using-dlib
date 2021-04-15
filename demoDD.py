import cv2
import dlib
from scipy.spatial import distance

def calculate_eyeRatio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    success, frame = cap.read()
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(grayImg)
    for face in faces:

        face_landmarks = dlib_facelandmark(grayImg, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	leftEye.append((x,y))
        	next_point = n+1
        	if n == 41:
        		next_point = 36
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	rightEye.append((x,y))
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_eyeRatio(leftEye)
        right_ear = calculate_eyeRatio(rightEye)

        eyeRatio = (left_ear+right_ear)/2
        eyeRatio = round(eyeRatio, 2)
        if eyeRatio < 0.26:
        	cv2.putText(frame, "DROWSY Alert!", (20,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 5)
        	cv2.putText(frame, "Don't Sleep!", (20,400), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 5)
        	print("Drowsy")
        print(eyeRatio)

    cv2.imshow("Are you Sleep!", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        break
cap.release()
cv2.destroyAllWindows()