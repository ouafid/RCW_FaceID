import cv2,numpy as np,face_recognition,os

#image path
path ='./images'

#global variables
image_list=[]#list des images
name_list=[]#list des noms

#recuperation des images d'un dossier 
mylist=os.listdir(path)
# print(mylist)

#chargement de l'image
for img in mylist:
    curImg=cv2.imread(os.path.join(path,img))
    image_list.append(curImg)
    imgName=os.path.splitext(img)[0]
    name_list.append(imgName)
    
#definir une fonction pour detecter les visage et extraire les caracteristiques
def findEncoding(img_list,imgNames_list):
    """_summary_
    definir une fonction pour detecter les visage et extraire les caracteristiques
    arg:
        img_list(list): list of BGR of images
        imgNames_list(list):list of images names
    """
    signatures_db=[]
    count=1
    for myImg,name in zip(img_list,imgNames_list):
        img=cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB)
        signature=face_recognition.face_encodings(img)[0]
        signature_class=signature.tolist()+[name]
        signatures_db.append(signature_class)
        print(f'{int((count/(len(img_list)))*100)}%extracted')
        count+=1
    signatures_db=np.array(signatures_db)
    np.save('FaceSignature_db.npy',signatures_db)
    print ('Signature_db stored')
    
def main():
    findEncoding(image_list,name_list)
    
if __name__=='__main__':
    main()