from torchvision.models.detection.rpn import AnchorGenerator
import torchvision

import torch
from torch.utils.data import Dataset, DataLoader

import os, json, cv2, numpy as np, matplotlib.pyplot as plt
import time

from references.visual import visualize

def get_model(num_keypoints, weights_path=None):

    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0)) # object with specified anchor sizes and aspect ratios.

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=None, # he detection head (the part responsible for object detection and keypoint prediction) will be initialized with random weights
                                                                   progress = True,
                                                                   weights_backbone='ResNet50_Weights.DEFAULT', #backbone network (ResNet-50) will be initialized with pre-trained weights from ImageNet
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   trainable_backbone_layers = 5,
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model


def inference(model,video_path,output_file,writer):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Transférer le modèle sur l'appareil cible
        model = model.to(device)

        # Définir le modèle en mode d'évaluation
        model.eval()

        # Définir le chemin d'accès à la vidéo
        video_path = video_path

        # Ouvrir la vidéo avec OpenCV
        cap = cv2.VideoCapture(video_path)

        # Lire la largeur et la hauteur de la vidéo
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Définir le codec et le fichier de sortie pour enregistrer la vidéo avec les prédictions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_file = output_file
        out = cv2.VideoWriter(out_file, fourcc, 20.0, (width, height))

        global_step = 0
        fc = 0
        fps = 0
        tfc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tfcc = 0

        print(f"Total frame count : {tfc}")

        #frames = np.empty((tfc, height, width, 3), dtype=np.uint8)
        #print("frame shape:",frames.shape)

        #i=0
        # Lire la vidéo image par image
        while tfcc < tfc:

            start_time = time.time()

            # Lire une image de la vidéo
            ret, frame = cap.read()

            # Si la fin de la vidéo est atteinte, sortir de la boucle
            if not ret:
                break

            # Convertir l'image en tenseur PyTorch
            image = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255
            #image = (frame.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
            image = image.to(device)
            #print("image shape", image.shape)

            # Effectuer une prédiction avec le modèle
            with torch.no_grad():
                output = model(image)

            # Extraire les prédictions de la sortie du modèle
            scores = output[0]['scores'].detach().cpu().numpy()
            high_scores_idxs = np.where(scores > 0.7)[0].tolist()
            post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

            keypoints = []
            for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])

            bboxes = []
            for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                bboxes.append(list(map(int, bbox.tolist())))

            # Visualiser les prédictions sur l'image
            frame = visualize(frame, bboxes, keypoints)
            #print("frame shape :", frame.shape)
            # Écrire l'image avec les prédictions dans le fichier de sortie
            out.write(frame)

            #frames[i] = frame
            #i +=1

            # Calculer le FPS
            fc += 1
            end_time = time.time()
            fps += 1/np.round(end_time - start_time, 3)
            if fc == 10:
                fps = int(fps / 10)
                tfcc += fc
                fc = 0
                per_com = int(tfcc / tfc * 100)
                print(f"Frames Per Second : {fps} || Percentage Parsed : {per_com}")
                start_time = end_time
            #print("tfcc :", tfcc)
            #print("tfc :", tfc)

            torch.cuda.empty_cache()

        #frames = frames.unsqueeze(0)
        #print("video tenseur shape:", frames.shape)
        #vid_tensor = torch.from_numpy(frames).float() 
        #writer.add_video('Inference video', vid_tensor, global_step=0, fps=4, walltime=None)

        # Libérer les ressources OpenCV
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def inference_realtime(model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transférer le modèle sur l'appareil cible
    model = model.to(device)

    # Définir le modèle en mode d'évaluation
    model.eval()

    # Ouvrir la vidéo avec OpenCV
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    cap.set(cv2.CAP_PROP_FPS, 36)

    frame_count = 0
    last_logged = time.time()

    # Lire la vidéo image par image
    while True:

        # Lire une image de la vidéo
        ret, frame = cap.read()

        # Si la fin de la vidéo est atteinte, sortir de la boucle
        if not ret:
            break

        # Convertir l'image en tenseur PyTorch
        image = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255
        #image = (frame.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        image = image.to(device)

        # Effectuer une prédiction avec le modèle
        with torch.no_grad():
            output = model(image)

        # Extraire les prédictions de la sortie du modèle
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(map(int, kp[:2])) for kp in kps])

        bboxes = []
        for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            bboxes.append(list(map(int, bbox.tolist())))

        # Visualiser les prédictions sur l'image
        frame = visualize(frame, bboxes, keypoints)

        cv2.imshow('Frame', frame)

        # Attendre 1 milliseconde pour permettre à l'utilisateur de fermer la fenêtre
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0

    # Libérer les ressources OpenCV
    cap.release()
    cv2.destroyAllWindows()