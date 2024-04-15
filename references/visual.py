import os, json, cv2, numpy as np, matplotlib.pyplot as plt

def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    
    fontsize = 5
    keypoints_classes_ids2names = {0: 'CH', 1: 'CM', 2:'CB'}

    img_width, img_height = image.shape[:2]

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    idx_nan =[]
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), tuple(kp), 5, (255,0,0), 4)
            image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            if kp == [np.nan,np.nan] :
                idx_nan = idx_nan.append(idx)
        if idx_nan == 0:
            image = cv2.line(image.copy(), kps[1], kps[2], (0,0,0), 1) 
        elif idx_nan == 1:
            image = cv2.line(image.copy(), kps[0], kps[2], (0,0,0), 1) 
        elif idx_nan == 2:
            image = cv2.line(image.copy(), kps[0], kps[1], (0,0,0), 1) 
        elif idx_nan == []:
            image = cv2.line(image.copy(), kps[0], kps[1], (0,0,0), 1) 
            image = cv2.line(image.copy(), kps[1], kps[2], (0,0,0), 1) 
        else:
            continue

    if image_original is None and keypoints_original is None:
        plt.figure(figsize=(40,40))
        plt.imshow(image)

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)
        
        for kps in keypoints_original:
            for idx, kp in enumerate(kps):
                image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 2)
                image_original = cv2.putText(image_original, " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)

    return image