import os, json, cv2, numpy as np, matplotlib.pyplot as plt


import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchsummary import summary
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import fiftyone as fo

import pandas as pd
import ast
import shutil
from PIL import Image
import json
import random
import time
import datetime
from torchvision.transforms import Normalize
import sys

from references import transforms, utils, engine, train
from references.utils import collate_fn
from references.engine import train_one_epoch, evaluate

from references import coco_format
from references.coco_format import change_name,fill_annotations_images, fill_json_annotations, split
from references.augmentation import train_transform, ClassDataset
from references.visual import visualize
from references.model import get_model,inference,inference_realtime

import argparse
import subprocess
import webbrowser


def main():

    writer = SummaryWriter('runs/Critac_Project')

    # Créer un parseur d'arguments
    parser = argparse.ArgumentParser(description='Initialiser les variables')

    # Ajouter des arguments au parseur
    parser.add_argument('mode', type=str, default='training', help='Mode d utilisation du code : data, training, test, inference')
    parser.add_argument('data_path', type=str, default='C:/Users/mcossin/Documents/object_detection/code_detection/keypoint_youtube-sebastien-2024-02-05/labeled-data', help='Chemin vers le fichiers labeled data du project deeplabcut')
    parser.add_argument('new_data_folder', type=str, default='', help='Chemin vers le fichier ou les données seront crée')
    parser.add_argument('--model', type=str, default=None, help='Chemin vers le model a tester ou pour l inference')
    parser.add_argument('--inference', type=str, default=None, help='Chemin vers le fichier ou la video a inferer provient')
    parser.add_argument('--realtime', type=int, default=None, help='specifie qu une inference en temps reel doit etre fait')

    # Parser les arguments
    args = parser.parse_args()

    # Initialiser les variables avec les valeurs spécifiées par l'utilisateur
    mode = args.mode
    labeled_data_path = args.data_path
    Coco_folder = args.new_data_folder
    if args.mode == 'test' :
        model_test = args.model
    if args.mode == 'inference' :
        model_to_inference = args.model
        inference_file = args.inference
        real_time = args.realtime

    # Afficher les valeurs des variables
    print(f'Tu as chosit le mode: {mode}')
    print(f'Chemin vers les données de deeplabcut: {labeled_data_path}')
    print(f'Chemin vers nouveaux avec le format COCO: {Coco_folder}')
    if args.mode == 'test' :
        print(f'Chemin vers le model a tester: {model_test}')
    if args.mode == 'inference':
        print(f'Chemin vers le model pour l inference {model_to_inference}')
        print(f'Chemin vers la video pour l inference {inference_file}')

    ############## Data preparation

    #labeled_data_path = r"C:\Users\mcossin\Documents\object_detection\code_detection\keypoint_youtube-sebastien-2024-02-05\labeled-data"
    #output_file_images = r"C:\Users\mcossin\Documents\object_detection\code_detection\keypoint_detection\dataset_exp\images"
    #output_file_annotations = r"C:\Users\mcossin\Documents\object_detection\code_detection\keypoint_detection\dataset_exp\annotations"
    
    #json_file = r"C:\Users\mcossin\Documents\object_detection\code_detection\keypoint_detection\dataset_exp\annotations\annotations.json"

    if mode == 'data':
        print("")
        print("Transformation des données provenant de Deeplabcut vers un format COCO")
        print("")

        if len(os.listdir(Coco_folder)) == 0:

            print("Création des dossirs images et annotations.")

            output_file_annotations = os.path.join(Coco_folder,'annotations')
            os.makedirs(output_file_annotations)

            jsonfile = os.path.join(output_file_annotations,'annotations.json')
            with open(jsonfile, "w") as json_file:
                json.dump({}, json_file)

            output_file_images = os.path.join(Coco_folder,'images')
            os.makedirs(output_file_images)

            #### change le nom des images pour simplifier le fichier json
            print("Change le nom des frames provenant des vidéos :")
            csv = change_name(output_file_images, labeled_data_path)

            index_keypoint = 0
            index_image = 0 

            #### remplis images et annotations pour le fichier json
            images,annotations = fill_annotations_images(csv,output_file_images,index_keypoint,index_image)

            #### colle les éléments du fichier json
            fill_json_annotations(images, annotations, output_file_annotations)

            #### split le json en 3 fichiers json pour le train, valid et test

            ## Choisir le bon fichier json
            #Fichier json avec l'annotation manuelle de l'attribut fait par Sebastien
            #json_file_group = r"C:\Users\mcossin\Documents\object_detection\code_detection\keypoint_detection\New_Dataset\annotations\annotations.json"
            
            # Choisis le pourcentage de découpage
            pourcent_train = 0.7
            pourcent_val = 0.15

            print("")
            print("Création des fichiers json au format COCO : ")
            ####
            split(jsonfile,output_file_annotations,pourcent_train,pourcent_val)   

            print("")
            print("Changer manuellement l'attribut group si ce n'est pas fait")
            print("Attribut a 0 par default")
            print("Pour modifier, il faut modifier le fichier annotations.json et ensuite relancer le code et entrer le fichier modifier directement dans la fonction split")

        else : 

            print("Les dossiers existent déja, seul les splits sont modifiés ")

            output_file_annotations = os.path.join(Coco_folder,'annotations')
            jsonfile = os.path.join(output_file_annotations,'annotations.json')
            output_file_images = os.path.join(Coco_folder,'images')

            pourcent_train = 0.7
            pourcent_val = 0.15

            ###### split les données avec le fichier modifiés annotations.json 
            split(jsonfile,output_file_annotations,pourcent_train,pourcent_val)   

    output_file_annotations = os.path.join(Coco_folder,'annotations')
    jsonfile = os.path.join(output_file_annotations,'annotations.json')
    output_file_images = os.path.join(Coco_folder,'images')

    csv = change_name(output_file_images, labeled_data_path)

    index_keypoint = 0
    index_image = 0 

    #### remplis images et annotations pour le fichier json
    images,annotations = fill_annotations_images(csv,output_file_images,index_keypoint,index_image)

    ############## Visualisation des données 

    visualisation = True

    if visualisation :
        print("")
        print("Visualisation des données :")

        annotations_path = jsonfile

        # https://docs.voxel51.com/user_guide/dataset_creation/datasets.html#loading-datasets-from-disk
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=output_file_images,
            labels_path=annotations_path,
        )

        session = fo.launch_app(dataset)
        session
    
    ############## Viusalisation des transformations

    #Coco_folder = r"C:\Users\mcossin\Documents\object_detection\code_detection\keypoint_detection\dataset_exp"
    visualisation_tranforms = True

    if visualisation_tranforms:

        dataset = ClassDataset(Coco_folder, "train.json", transform=train_transform(), demo=True)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

        iterator = iter(data_loader)
        batch = next(iterator)

        #print("Original targets:\n", batch[3], "\n\n")
        #print("Transformed targets:\n", batch[1])

        image = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
        bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

        keypoints = []
        for kps in batch[1][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
            keypoints.append([kp[:2] for kp in kps])

        #image_original = (batch[2][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
        #bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

        keypoints_original = []
        for kps in batch[3][0]['keypoints'].detach().cpu().numpy().astype(np.int32).tolist():
            keypoints_original.append([kp[:2] for kp in kps])

        image = visualize(image, bboxes, keypoints)

        # Convert the image tensor to the expected format (NCHW)
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1)))
        image_tensor = image_tensor.unsqueeze(0)
        print("image shape :",image_tensor.shape)

        writer.add_images('Images avec keypoints', image_tensor, global_step=0)

    ############## Training et Validation

    ## Creation data loader
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 4

    dataset_train = ClassDataset(Coco_folder,"train.json", transform=train_transform(), demo=False)
    dataset_val = ClassDataset(Coco_folder, "valid.json", transform=None, demo=False)
    dataset_test = ClassDataset(Coco_folder,"test.json", transform=None, demo=False)

    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

    checkpoint = False
    if checkpoint == True:
        checkpoint_path= r"C:\Users\mcossin\Documents\object_detection\code_detection\keypoint_detection\Experiences\Exp11\Exp11\model_29_baseline.pth"
        model = get_model(num_keypoints=3)
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.to(device)
        #model = get_model(num_keypoints = 3, weights_path=checkpoint_path)
        num_epochs = 10
        print("Checkpoint :",checkpoint_path)

    else : 
        model = get_model(num_keypoints = 3)
        num_epochs = 2

    start_epoch = 0

    ## Hyperparametres
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, betas = (0.9, 0.999),eps = 1e-8, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
    # step_size (int): Period of learning rate decay.
    # gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.

    ## Dossier d'enregistremnt de l'entrainement
    num_exp = '18'
    project_path = os.path.dirname(Coco_folder)
    output_dir = os.path.join(project_path,'Experiences','Exp')
    output_dir = output_dir + num_exp
    #output_dir = r"C:\Users\mcossin\Documents\object_detection\code_detection\keypoint_detection\Experiences\Exp" + num_exp
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_path = os.path.join(output_dir,'loss.csv')
    
    ## Debut de l'entrainement
    if mode == 'training' :

        print("Start training")
        start_time = time.time()
        for epoch in range(start_epoch, num_epochs):
            metric_logger,loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1,csv_path = csv_path,writersum = writer)
            lr_scheduler.step()

            writer.add_scalar('Training loss epoch', loss, epoch)
            #enregistre toutes les 5 epochs
            if (epoch + 1) % 5 == 0:

                torch.save(model.state_dict(), os.path.join(output_dir, f"model_{epoch}_baseline.pth"))

            # evaluate after every epoch
            coco_evaluator = evaluate(model, data_loader_val, device=device)

            #writer.add_scalar('Validation loss epoch', loss_val, epoch)

            keypoints_eval_info = coco_evaluator.coco_eval['keypoints'].stats.tolist()
            bbox_eval_info = coco_evaluator.coco_eval['bbox'].stats.tolist()

            writer.add_scalar('Validation Keypoints mAP progression', keypoints_eval_info[0], epoch)
            writer.add_scalar('Validation Keypoints mAR progression', keypoints_eval_info[5], epoch)

            writer.add_scalar('Validation bbox mAP progression', bbox_eval_info[0], epoch)
            writer.add_scalar('Validation bbox mAR progression', bbox_eval_info[6], epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")


    ## Debut du test
    if mode == 'test':

        #model_to_test = os.path.join(output_dir,'model_19_baseline.pth')
        model = get_model(num_keypoints = 3, weights_path=model_test)
        model.to(device)
        print("\nStart testing")
        evaluate(model, data_loader_test, device=device)

        ## Visualisation 
        global_step = 0

        for i, (images, targets) in enumerate(data_loader_test):
            images = list(image.to(device) for image in images)

            with torch.no_grad():
                output = model(images)

            for j, image in enumerate(images):
                image = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

                scores = output[j]['scores'].detach().cpu().numpy()
                high_scores_idxs = np.where(scores > 0.9)[0].tolist()
                post_nms_idxs = torchvision.ops.nms(output[j]['boxes'][high_scores_idxs], output[j]['scores'][high_scores_idxs], 0.3).cpu().numpy()

                keypoints = []
                for kps in output[j]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                    keypoints.append([list(map(int, kp[:2])) for kp in kps])

                bboxes = []
                for bbox in output[j]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                    bboxes.append(list(map(int, bbox.tolist())))

                print(f"Predictions for Batch {i + 1}, Image {j + 1}:")

                image = visualize(image, bboxes, keypoints)

                # Convert the image tensor to the expected format (NCHW)
                image_tensor = torch.from_numpy(image.transpose((2, 0, 1)))
                image_tensor = image_tensor.unsqueeze(0)
                print("image shape :",image_tensor.shape)

                writer.add_images('Test Visualisation', image_tensor, global_step=global_step)
                global_step +=1

    if mode == 'inference':
        #model_to_test = os.path.join(output_dir,'model_19_baseline.pth')
        model = get_model(num_keypoints = 3, weights_path=model_to_inference)

        if real_time == 1:

            inference_realtime(model)

            print ("\nVideo inference en tenps reel")

        else :

            #video_path = r"C:\Users\mcossin\Documents\object_detection\Data\youtube_video\test\video-youtube-2-test.mp4"
            video_path = inference_file

            video_name = os.path.basename(video_path)

            inference_name = 'inference_' + video_name
            #output_file = os.path.join(r"C:\Users\mcossin\Documents\object_detection\code_detection\keypoint_detection\Inference",'video-youtube-2-inference.mp4')
            output_file = os.path.join(project_path, 'Inference', inference_name)

            inference(model,video_path,output_file,writer)

            print ("\nVideo inference enregisté dans ce chemin", output_file)

    else: 
        print("Tu t'es trompé de mode ! \nLes modes possibles sont training, test et inference.")

    ### Lance Tensorboard automatiquement
    logdir = 'runs/Critac_Project'

    # Lancer TensorBoard en arrière-plan
    process = subprocess.Popen(['tensorboard', '--logdir', logdir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Ouvrir un navigateur web pour afficher TensorBoard
    url = f'http://localhost:6006/?logdir={logdir}'
    webbrowser.open(url)

    process.wait()

    writer.close()
if __name__ == "__main__":
    main()