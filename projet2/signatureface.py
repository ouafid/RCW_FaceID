import cv2
import numpy as np
import face_recognition
import os

# Image database path
path = './dataImg'

# Global variables
image_list = []  # List of images
name_list = []  # List of image names

# Grab all images from the folder
my_list = os.listdir(path)

# Load images
for img in my_list:
    if os.path.splitext(img)[1].lower() in ['.jpg', '.png', '.jpeg']:
        cur_img = cv2.imread(os.path.join(path, img))
        image_list.append(cur_img)
        img_name = os.path.splitext(img)[0]
        name_list.append(img_name)

# Define a function to detect face and extract features
def find_encodings(img_list, img_name_list):
    signatures_db = []
    count = 1
    for my_img, name in zip(img_list, img_name_list):
        if my_img is None:
            print(f"Unable to read image: {name}")
            continue  # Skip to the next image if unable to read
        img = cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB)
        # Detect faces in the image
        face_locations = face_recognition.face_locations(img)
        if face_locations:
            # If faces are detected, extract face encodings
            signature = face_recognition.face_encodings(img, face_locations)[0]
            signature_class = signature.tolist() + [name]
            signatures_db.append(signature_class)
        else:
            print(f"No face detected in image: {name}")
        print(f"{int((count / len(img_list)) * 100)}% extracted")
        count += 1

    signatures_db = np.array(signatures_db)
    np.save('Facedb.npy', signatures_db)
    print('Signature database stored')

def main():
    find_encodings(image_list, name_list)

if __name__ == '__main__':
    main()

