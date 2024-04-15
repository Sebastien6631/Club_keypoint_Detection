import albumentations as A 
from torchvision.transforms import functional as F

import torch
from torch.utils.data import Dataset, DataLoader

import os, json, cv2, numpy as np, matplotlib.pyplot as plt

class ClassDataset(Dataset):
    def __init__(self, root, jsonfile, transform=None, demo=False):                
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.annotations_file = os.path.join(root, "annotations", jsonfile)  # Chemin vers le fichier JSON des annotations

        with open(self.annotations_file) as f:
            self.annotations = json.load(f)

        self.imgs_files = [images["file_name"] for images in self.annotations["images"]]
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        for image_info in self.annotations["images"]:
            if image_info["file_name"] == self.imgs_files[idx]:
                image_id = image_info["id"]
                break

        annotations = []
        for annotation_info in self.annotations["annotations"]:
            if annotation_info["image_id"] == image_id:
                annotations.append(annotation_info)

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)        
        
        ### taille des images
        img_height, img_width = img_original.shape[:2]

        ### Recupere les boites dans le format coco x,y,w,h et les keypoints
        bboxes_original = []
        keypoints_originals = []
        for annotation in annotations:

            bbox = annotation["bbox"]
            bboxes_original.append(bbox)

            kp = annotation["keypoints"]
            keypoints_originals.append(kp)
        
        #### Crée une liste avec les keypoints dans le format coco pour la transformation
        #### et crée une liste dans le format x1, y1, x2, y2 pour la visualisation
        bboxes_original_xywh = []
        bboxes_original_xy = []
        for bbox in bboxes_original:
            x1, y1, w, h = bbox
            bboxes_original_xywh.append([x1,y1,w,h])
            bboxes_original_xy.append([min(max(x1,0),img_width), min(max(y1,0),img_height), min(max(w+x1,0),img_width), min(max(h+y1,0),img_height)])

        #### Convertis le format coco des annotations keypoints [x,y,visibility, x,y,visibility, x,y,visibility] 
        ####  au format [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
        #### Une liste avec les NaN, ainsi que leurs indices et une liste sans les NaN pour la transformations
        keypoints_original = []
        keypoints_original_with_nan = []
        nan_indices = []  
        for i in range(len(keypoints_originals)):
            keypoints = keypoints_originals[i]
            keypoints_list = []
            keypoints_list_with_nan = []
            for j in range(0, len(keypoints), 3):
                if np.isnan(keypoints[j]) or np.isnan(keypoints[j+1]):
                    nan_indices.append([i, j//3])  # Ajouter l'indice au format (indice de l'objet, indice du keypoint)
                    keypoints_list_with_nan.append([np.nan, np.nan, np.nan])
                else:
                    keypoints_list.append([keypoints[j], keypoints[j+1], keypoints[j+2]])
                    keypoints_list_with_nan.append([keypoints[j], keypoints[j+1], keypoints[j+2]])
            keypoints_original.append(keypoints_list)
            keypoints_original_with_nan.append(keypoints_list_with_nan)

        # All objects are glue tubes
        bboxes_labels_original = ['Club' for _ in bboxes_original_xy]            

        if self.transform:  

            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints            
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format            
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            
            keypoints_original_flattened = []
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            
            # Apply augmentations
            transformed = self.transform(image=img_original, bboxes=bboxes_original_xywh, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            
            img = transformed['image']
            bboxes = transformed['bboxes']
            
            #### Mets le format de la bounding box x,y,x,y
            bboxes_xy = []
            for bbox in bboxes:
                x1, y1, w, h = bbox
                x1 = round(max(min(x1, img_width), 0), 2)
                y1 = round(max(min(y1, img_height), 0), 2)
                x2 = round(max(min(x1 + w, 0), img_width), 2)
                y2 = round(max(min(y1 + h, 0), img_height), 2)
                bboxes_xy.append([x1, y1, x2, y2])
                
            keypoints_transformed = transformed['keypoints']
            
            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            # Divertly Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints

            #### Remplis avec les NaN et les keypoints dans les listes
            keypoint = []
            keypoint_idx = 0
            for i in range(len(keypoints_original_with_nan)):
                obj_keypoints = []
                for j in range(len(keypoints_original_with_nan[i])):
                    if [i, j] in nan_indices:  # Vérifie si cet indice était un NaN
                        obj_keypoints.append(np.nan)
                        obj_keypoints.append(np.nan)
                        obj_keypoints.append(np.nan)
                    else:
                        if len(keypoints_transformed) <= keypoint_idx:
                            obj_keypoints.append(np.nan)
                            obj_keypoints.append(np.nan)
                            obj_keypoints.append(np.nan)
                        else :
                            obj_keypoints.extend(keypoints_transformed[keypoint_idx])
                            obj_keypoints.append(2)
                        keypoint_idx += 1

                keypoint.append(obj_keypoints)

            #### Reconstruis la liste de liste
            keypoints = []                   
            for i in range(len(keypoint)):
                obj_keypoint = []
                for j in range(0,len(keypoint[i]),3):
                    obj_keypoint.append([keypoint[i][j], keypoint[i][j+1], keypoint[i][j+2]])
                keypoints.append(obj_keypoint)

       
        else:
            img, bboxes_xy, keypoints = img_original, bboxes_original_xy, keypoints_original_with_nan        
        
        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor(bboxes_xy, dtype=torch.float32)       
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are glue tubes
        target["image_id"] = image_id
        try:
            target["area"] = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        except IndexError:
            print(f"IndexError: Image at index {image_id} has no annotations")
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)        
        img = F.to_tensor(img)
        
        bboxes_original = torch.as_tensor(bboxes_original_xy, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64) # all objects are glue tubes
        target_original["image_id"] = image_id
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original_with_nan, dtype=torch.float32)        
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs_files)



def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=0.5),  # Random change of brightness & contrast
            A.VerticalFlip(p=0.5),  # Randomly flips the input vertically
            A.CenterCrop (700, 700, always_apply=False, p=1.0),
            A.RandomGamma(gamma_limit=(50, 150), p=0.5),
            A.GaussNoise(var_limit=(10, 50), mean=0, p=0.5),
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'),  # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='coco', label_fields=['bboxes_labels'])  # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )