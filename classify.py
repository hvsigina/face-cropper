import face_recognition
import os

include_image = face_recognition.load_image_file("include.jpg")
exclude_image = face_recognition.load_image_file("exclude.jpg")
output_path = "./output"

files = os.listdir(output_path)

for file in files:

    include_encoding = face_recognition.face_encodings(include_image)[0]
    exclude_encoding = face_recognition.face_encodings(exclude_image)[0]
    
    unknown_path = output_path+"/"+file
    
    unknown_image = face_recognition.load_image_file(unknown_path)
    if len(face_recognition.face_encodings(unknown_image))<1:
        print(unknown_path)
        if os.path.exists(unknown_path):
            os.remove(unknown_path)
        continue
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    ex_results = face_recognition.compare_faces([exclude_encoding], unknown_encoding,0.8)
    in_results = face_recognition.compare_faces([include_encoding], unknown_encoding,0.8)
    
    if not in_results[0] and  ex_results[0]:
        print(unknown_path)
        if os.path.exists(unknown_path):
            os.remove(unknown_path)




