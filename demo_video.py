import cv2
import os
import face_recognition
from scipy.datasets import face
from face_cropper import FaceCropper

#face_cropper = FaceCropper()
face_cropper = FaceCropper(min_face_detector_confidence=0.1, face_detector_model_selection=FaceCropper.LONG_RANGE, landmark_detector_static_image_mode=FaceCropper.TRACKING_MODE, min_landmark_detector_confidence=0.1)

border_size = 0
img_res = 256

#camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Capture video with default camera (Use DSHOW API for reading to avoid SourceReader warning)
camera = cv2.VideoCapture('demo1.mp4')  # Read video from specified path
i=0
output_path = "D:/NEW/photos/face-cropper/output"


while True:
    read_successful, image_bgr = camera.read()
    if not read_successful: raise RuntimeError('Image could not be read!')
    
    faces_rgb = face_cropper.get_faces(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),remove_background=False, correct_roll=True)
    #faces_rgb = face_cropper.get_faces_debug(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    
    i+=1
    if not faces_rgb:
        print("No faces detected!")
    else:
        cv2.imshow('Image', image_bgr)
        for face_id, face_rgb in enumerate(faces_rgb):
            cv2.imshow('Face {0}'.format(face_id), cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
            outimg = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            outname = str(i)+"_mod_"+str(face_id)+".jpg"
            outpath = os.path.join(output_path , outname)
            outimg= cv2.copyMakeBorder(outimg,border_size,border_size,border_size,border_size,cv2.BORDER_CONSTANT,value=[0,0,0])
            height, width, _ = outimg.shape
            ratio=1
            #cv2.imwrite(outpath+"_before.jpg",outimg)
            if height>width:
                ratio = img_res/width
            else:
                ratio = img_res/height
            width = int(width*ratio)
            height =int(height*ratio)
            outimg = cv2.resize(outimg, (width,height))
            
            cv2.imwrite(outpath,outimg)

    if cv2.pollKey() != -1:  # User pressed key
        camera.release()
        cv2.destroyAllWindows()
        break
   