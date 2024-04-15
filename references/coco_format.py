import os
import pandas as pd
import ast
import shutil
from PIL import Image
import json
import random
import numpy as np
import matplotlib.pyplot as plt

def get_csv(labeled_data_path):

    # Recupere le path des csv
    csv_path = []
    for item in os.listdir(labeled_data_path):
    
        folder_path = os.path.join(labeled_data_path, item)

        if os.path.isdir(folder_path):
            
            if not item.endswith("_labeled"):

                for idx,it in enumerate(os.listdir(folder_path)):

                    if it.endswith("csv"):

                        csv_path.append(os.path.join(folder_path, it))
    
    # Choix des csv
    csv0 = pd.read_csv(csv_path[0])
    csv1 = pd.read_csv(csv_path[1])
    csv3 = pd.read_csv(csv_path[3])

    # Supprime les 3 premieres lignes car 
    csv0 = csv0.drop(csv0.index[:3])
    csv1 = csv1.drop(csv1.index[:3])
    csv3 = csv3.drop(csv3.index[:3])

    # Concatene les csv
    csv = pd.concat([csv0, csv1, csv3], ignore_index=True)

    return csv



def fill__images_file(dataframe, dossier_images_source, dossier_images_destination,num_csv, idx):
    
    if not os.path.exists(dossier_images_destination):
        os.makedirs(dossier_images_destination)

    for index, row in dataframe.iterrows():
        
        #if index < 3:
        #    continue

        nom_fichier = row['Unnamed: 2']
        video_origin = row['Unnamed: 1']

        if pd.isna(nom_fichier):
            continue

        chemin_source = os.path.join(dossier_images_source, nom_fichier)

        nouveau_nom_fichier = f"{video_origin}_{index}.png" #+ idx}.png"

        chemin_destination = os.path.join(dossier_images_destination, nouveau_nom_fichier)

        shutil.copy(chemin_source, chemin_destination)

        dataframe.at[index, 'Unnamed: 2'] = nouveau_nom_fichier

    idx = index + idx
    #idx = index-3 + idx

    dataframe.to_csv(os.path.join(dossier_images_destination, f"annotations_{num_csv}.csv"), index=False)

    return(dataframe)

def add_area_to_annotations(annotations):
    for anno in annotations:
        x1, y1, w, h = anno['bbox']
        area = w * h
        anno['area'] = area
    return annotations


def change_name(dossier_images_destination, labeled_data_path):
    idx=0
    csv = get_csv(labeled_data_path)
    group_counts = csv.groupby("Unnamed: 1").size()
    print("group :", group_counts)
    lengths = group_counts.tolist()
    group_indices = [group for _, group in csv.groupby("Unnamed: 1")]

    dfs = []
    for i, group in enumerate(group_indices):
        folder_path = os.path.join(labeled_data_path, group_counts.index[i])
        df = fill__images_file(group, folder_path, dossier_images_destination, i, idx)
        idx += lengths[i]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df

def form_bbox(clubhead, clubmiddle, clubbottom, pixel_box):

    if (not np.isnan(clubhead[0]) and not np.isnan(clubhead[1])) and (not np.isnan(clubmiddle[0]) and not np.isnan(clubmiddle[1])) and (not np.isnan(clubbottom[0]) and not np.isnan(clubbottom[1])):
        
        bbox_orig = [min(clubhead[0], clubmiddle[0], clubbottom[0]),
                min(clubhead[1], clubmiddle[1], clubbottom[1]),
                max(clubhead[0], clubmiddle[0], clubbottom[0]),
                max(clubhead[1], clubmiddle[1], clubbottom[1])]
        
        width = round(bbox_orig[2] - bbox_orig[0])
        height = round(bbox_orig[3] - bbox_orig[1])

        bbox = [max(round(bbox_orig[0] - pixel_box),0), 
                max(round(bbox_orig[1] - pixel_box),0), 
                max(round(width + 2*pixel_box),0), 
                max(round(height + 2*pixel_box),0)]
        
    elif (not np.isnan(clubhead[0]) and not np.isnan(clubhead[1])) and (not np.isnan(clubmiddle[0]) and not np.isnan(clubmiddle[1])):

        bbox_orig = [min(clubhead[0], clubmiddle[0]),
                min(clubhead[1], clubmiddle[1]),
                max(clubhead[0], clubmiddle[0]),
                max(clubhead[1], clubmiddle[1])]
        
        width = round(bbox_orig[2] - bbox_orig[0])
        height = round(bbox_orig[3] - bbox_orig[1])

        bbox = [max(round(bbox_orig[0] - pixel_box),0), 
                max(round(bbox_orig[1] - pixel_box),0), 
                max(round(width + 2*pixel_box),0), 
                max(round(height + 2*pixel_box),0)]
            
    elif (not np.isnan(clubmiddle[0]) and not np.isnan(clubmiddle[1])) and (not np.isnan(clubbottom[0]) and not np.isnan(clubbottom[1])):

        bbox_orig = [min(clubmiddle[0], clubbottom[0]),
                min(clubmiddle[1], clubbottom[1]),
                max(clubmiddle[0], clubbottom[0]),
                max(clubmiddle[1], clubbottom[1])]

        width = round(bbox_orig[2] - bbox_orig[0])
        height = round(bbox_orig[3] - bbox_orig[1])

        bbox = [max(round(bbox_orig[0] - pixel_box),0), 
                max(round(bbox_orig[1] - pixel_box),0), 
                max(round(width + 2*pixel_box),0), 
                max(round(height + 2*pixel_box),0)] 
               
    elif (not np.isnan(clubhead[0]) and not np.isnan(clubhead[1])) and (not np.isnan(clubbottom[0]) and not np.isnan(clubbottom[1])):

        bbox_orig = [min(clubhead[0], clubbottom[0]),
                min(clubhead[1], clubbottom[1]),
                max(clubhead[0], clubbottom[0]),
                max(clubhead[1], clubbottom[1])]

        width = round(bbox_orig[2] - bbox_orig[0])
        height = round(bbox_orig[3] - bbox_orig[1])

        bbox = [max(round(bbox_orig[0] - pixel_box),0), 
                max(round(bbox_orig[1] - pixel_box),0), 
                max(round(width + 2*pixel_box),0), 
                max(round(height + 2*pixel_box),0)]
    else:

        if (not np.isnan(clubhead[0]) and not np.isnan(clubhead[1])):

            bbox_orig = [clubhead[0] - pixel_box, clubhead[1] - pixel_box, clubhead[0] + pixel_box, clubhead[1] + pixel_box]

            width = round(bbox_orig[2] - bbox_orig[0])
            height = round(bbox_orig[3] - bbox_orig[1])
            bbox = [max(round(bbox_orig[0] - pixel_box),0), 
                max(round(bbox_orig[1] - pixel_box),0), 
                max(round(width + 2*pixel_box),0), 
                max(round(height + 2*pixel_box),0)]
        
        elif (not np.isnan(clubmiddle[0]) and not np.isnan(clubmiddle[1])):

            bbox_orig = [clubmiddle[0] - pixel_box, clubmiddle[1] - pixel_box, clubmiddle[0] + pixel_box, clubmiddle[1] + pixel_box]

            width = round(bbox_orig[2] - bbox_orig[0])
            height = round(bbox_orig[3] - bbox_orig[1])
            bbox = [max(round(bbox_orig[0] - pixel_box),0), 
                max(round(bbox_orig[1] - pixel_box),0), 
                max(round(width + 2*pixel_box),0), 
                max(round(height + 2*pixel_box),0)]
            
        elif (not np.isnan(clubbottom[0]) and not np.isnan(clubbottom[1])):

            bbox_orig = [clubbottom[0] - pixel_box, clubbottom[1] - pixel_box, clubbottom[0] + pixel_box, clubbottom[1] + pixel_box]

            width = round(bbox_orig[2] - bbox_orig[0])
            height = round(bbox_orig[3] - bbox_orig[1])
            bbox = [max(round(bbox_orig[0] - pixel_box),0), 
                max(round(bbox_orig[1] - pixel_box),0), 
                max(round(width + 2*pixel_box),0), 
                max(round(height + 2*pixel_box),0)]  
            
        else :
            bbox = [0,0,0,0]
            
    return bbox


def fill_annotations_images(dataframe,folder_images,index_keypoint,index_image):
    annotations = []
    images = []
    index_key = index_keypoint
    index_image = index_image


    for index, row in dataframe.iterrows():
        Club = True
        
        ################################### Fill images part 
        image_name = row['Unnamed: 2']
        image_id = index  +index_image
        #image_id = index - 3 +index_image
        image_path = os.path.join(folder_images,image_name)
        img = Image.open(image_path)
        width, height = img.size

        ################################### Fill annotations part 
        ## Si entrainement sur une video avec plus de 3 clubs, changer ce parametre :
        keypoints_ranges = [(3, 8), (9, 14), (15, 20)]

        for start, end in keypoints_ranges:

            keypoints = []
            annotation = {}

            annotation = {"id": index_key, "image_id": image_id, "category_id": 1}

            for j in range(start, end, 2):
                if  pd.notna(row.iloc[j]) and pd.notna(row.iloc[j + 1]):
                    x_value = float(row.iloc[j])
                    y_value = float(row.iloc[j + 1])
                    keypoints.extend([x_value, y_value, 2])  # Assuming all keypoints are labeled
                else:
                    keypoints.extend([np.nan, np.nan, np.nan])

            clubhead = (keypoints[0],keypoints[1])
            clubmiddle = (keypoints[3],keypoints[4])
            clubbottom = (keypoints[6],keypoints[7])

            if clubhead ==(np.nan,np.nan) and clubmiddle ==(np.nan,np.nan) and clubbottom ==(np.nan,np.nan):
                ## Enleve les images avec seulement des annotations vides 
                ## c'est a dire avec moins de 3 clubs 
                print("image_id :", image_id)
                print("image_name :",image_name)
                Club = False

            else :
                annotation["keypoints"] = keypoints
                
                bbox = form_bbox(clubhead, clubmiddle, clubbottom, 7) ## rajoute 10 pixels sur les cotés

                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[1] < 0:
                    bbox[1] = 0
                if bbox[0] +  bbox[2] > width:
                    bbox[2] = width - bbox[0]
                if bbox[1] +  bbox[3] > height:
                    bbox[3] = height - bbox[1]
                #print(bbox)
                annotation["bbox"] = bbox
                
                annotation["iscrowd"]=0
                #if sum(1 for k in range(len(keypoints)) if (keypoints[k] and keypoints[k] != 2)) == 2 or sum(1 for k in range(len(keypoints)) if (keypoints[k] and keypoints[k] != 2)) == 0 :
                #    annotation["iscrowd"]=0
                #else:
                #    annotation["iscrowd"]=1

                annotation["num_keypoints"] = sum(1 for k in range(len(keypoints)) if (keypoints[k] and keypoints[k] != 2))

                annotations.append(annotation)

                index_key +=1

        if Club:
            image = {}
            image = {"id": image_id,"file_name" : image_name, "width":width, "height":height, "url": image_path,"group" : 0}
            
            images.append(image)

        Club = True

    return images, annotations

def fill_json_annotations(images, annotations, output_file):
    info = [
    {
        "description": "Critac Project Dataset",
        "version": "1.0",
        "year": 2024,
        "contributor": "Sebastien",
        "date_created": "2024/02/27"
    }
    ]

    categories = [
    {
        "supercategory": "Object",
        "id": 1,
        "name": "Club",
        "keypoints": [
            "Clubhead","Clubmiddle","Clubbottom"
        ],
        "skeleton": [
            [0,1],[1,2],[0,2]
        ]
    }
    ]   

    data = {
    "info": info[0],
    "categories": categories, 
    "images": images,
    "annotations": annotations,
    }

    output_file = output_file
    json_file_path = os.path.join(output_file,'annotations.json')

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def split(json_path,output,pourcent_train,pourcent_val):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    ## Enleve les images sans annotations
    image_ids = set(img['id'] for img in data["images"])
    filtered_annotations = [annotation for annotation in data["annotations"] if annotation['image_id'] in image_ids]
    data["annotations"] = filtered_annotations

    # enleve les nan
    for anno in data["annotations"]:
        keypoints = anno["keypoints"]
        for idx in range(0,len(keypoints),3):
            if np.isnan(keypoints[idx]):
                keypoints[idx] = 0.5
                keypoints[idx+1] = 0.5
                keypoints[idx+2] = 0
        anno["keypoints"] = keypoints

    # calcul le area
    data["annotations"] = add_area_to_annotations(data["annotations"])

    random.shuffle(data["images"])

    total_images = len(data["images"])

    group1_images = [img for img in data["images"] if img["group"] == 1]
    group1_count = len(group1_images)
    print("Nombre de frame avec l'attribut group à 1 :", group1_count)

    group0_images = [img for img in data["images"] if img["group"] == 0]
    group0_count = len(group0_images)
    print("Nombre de frame avec l'attribut group à 0 :", group0_count)

    group1_train_size = int(group1_count * pourcent_train)
    group1_valid_size = int(group1_count * pourcent_val)

    train_size = int(total_images * pourcent_train)
    valid_size = int(total_images * pourcent_val)

    #train_data = {"categories": data["categories"], "info": data["info"], "images": data["images"][:train_size], "annotations": []}
    #valid_data = {"categories": data["categories"], "info": data["info"], "images": data["images"][train_size:train_size+valid_size], "annotations": []}
    #test_data = {"categories": data["categories"], "info": data["info"], "images": data["images"][train_size+valid_size:], "annotations": []}

    train_data = {"categories": data["categories"], "info": data["info"], "images": [], "annotations": []}
    valid_data = {"categories": data["categories"], "info": data["info"], "images": [], "annotations": []}
    test_data = {"categories": data["categories"], "info": data["info"], "images": [], "annotations": []}

    # Ajouter les images group1 aux données de train et de validation
    train_data["images"].extend(group1_images[:group1_train_size])
    valid_data["images"].extend(group1_images[group1_train_size:group1_train_size + group1_valid_size])
    test_data["images"].extend(group1_images[group1_train_size + group1_valid_size:])

    # Ajouter le reste des images aux données de train, validation et test
    non_group1_images = [img for img in data["images"] if img["group"] != 1]
    train_data["images"].extend(non_group1_images[:train_size - group1_train_size])
    valid_data["images"].extend(non_group1_images[train_size - group1_train_size:train_size + valid_size - group1_train_size - group1_valid_size])
    test_data["images"].extend(non_group1_images[train_size + valid_size - group1_train_size - group1_valid_size:])

    group1_train = [img for img in train_data["images"] if img["group"] == 1]
    group1_valid = [img for img in valid_data["images"] if img["group"] == 1]
    group1_test = [img for img in test_data["images"] if img["group"] == 1]
    group0_train = [img for img in train_data["images"] if img["group"] == 0]
    group0_valid = [img for img in valid_data["images"] if img["group"] == 0]
    group0_test = [img for img in test_data["images"] if img["group"] == 0]

    print("Nombre de frame avec l'attribut group à 1 dans le split train :", len(group1_train))
    print("Nombre de frame avec l'attribut group à 1 dans le split valid :", len(group1_valid))
    print("Nombre de frame avec l'attribut group à 1 dans le split test :", len(group1_test))
    print("Nombre de frame avec l'attribut group à 0 dans le split train :", len(group0_train))
    print("Nombre de frame avec l'attribut group à 0 dans le split valid :", len(group0_valid))
    print("Nombre de frame avec l'attribut group à 0 dans le split test :", len(group0_test))

    # Nombre de frames avec l'attribut group à 1 dans chaque split
    group1_train = len(group1_train)
    group1_valid = len(group1_valid)
    group1_test = len(group1_test)

    # Nombre de frames avec l'attribut group à 0 dans chaque split
    group0_train = len(group0_train)
    group0_valid = len(group0_valid)
    group0_test = len(group0_test)

    plt.figure(figsize=(10, 6))

    # Tracing des données pour le split train
    plt.bar('Train group1', group1_train, color='r', label='Group 1')
    plt.bar('Train group0', group0_train, color='b', label='Group 0')

    # Tracing des données pour le split valid
    plt.bar('Valid group1', group1_valid, color='r')
    plt.bar('Valid group0', group0_valid, color='b')

    # Tracing des données pour le split test
    plt.bar('Test group1', group1_test, color='r')
    plt.bar('Test group0', group0_test, color='b')

    plt.xlabel('Split')
    plt.ylabel('Nombre de frames')
    plt.title('Histogramme des données')

    histogram_filename = 'Group_histogram.png'
    histogram_filepath = os.path.join(output, histogram_filename)
    plt.savefig(histogram_filepath)

    for annotation in data["annotations"]:
        if annotation["image_id"] in [image["id"] for image in train_data["images"]]:
            train_data["annotations"].append(annotation)
        elif annotation["image_id"] in [image["id"] for image in valid_data["images"]]:
            valid_data["annotations"].append(annotation)
        else:
            test_data["annotations"].append(annotation)

    for img_group in train_data["images"]:
        if img_group["group"] == 1:
            img_id = img_group["id"]
            annotations = [anno for anno in train_data["annotations"] if anno["image_id"] == img_id]
            for anno in annotations:
                anno["iscrowd"] = 1
            
    for img_group in valid_data["images"]:
        if img_group["group"] == 1:
            img_id = img_group["id"]
            annotations = [anno for anno in valid_data["annotations"] if anno["image_id"] == img_id]
            for anno in annotations:
                anno["iscrowd"] = 1

    for img_group in test_data["images"]:
        if img_group["group"] == 1:
            img_id = img_group["id"]
            annotations = [anno for anno in test_data["annotations"] if anno["image_id"] == img_id]
            for anno in annotations:
                anno["iscrowd"] = 1

    if not os.path.exists(output):
        os.makedirs(output)

    with open(os.path.join(output, "train.json"), "w") as train_json_file:
        json.dump(train_data, train_json_file, indent=4)

    with open(os.path.join(output, "valid.json"), "w") as valid_json_file:
        json.dump(valid_data, valid_json_file, indent=4)

    with open(os.path.join(output, "test.json"), "w") as test_json_file:
        json.dump(test_data, test_json_file, indent=4)