import torch
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

st = time.time()


CHECKPOINT_PATH ='sam_vit_h_4b8939.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

total_pores_area_px = 0
labels = {0: u'background', 1: u'pore'}

df_porosity = pd.DataFrame(columns = ['Sample #', 'Porosity, %'])


def sam_yolo(IMAGE_PATH):
    img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_predictor = SamPredictor(sam)

    model = YOLO('best_s.pt')
    results = model.predict(task="detect", source=IMAGE_PATH, conf=0.25)

    try:
        predicted_boxes = results[0].boxes
        transformed_boxes = mask_predictor.transform.apply_boxes_torch(predicted_boxes.xyxy, img_rgb.shape[:2])
        mask_predictor.set_image(img_rgb)
        masks, scores, logits = mask_predictor.predict_torch(boxes = transformed_boxes, multimask_output=False, point_coords=None, point_labels=None)
    except:
        print("No pores found")
        masks = None
        predicted_boxes = None
    
    return img_rgb, masks, predicted_boxes


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = 1
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 2, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 2, txt_color, thickness=tf, lineType=cv2.LINE_AA)


def draw_bb(img_rgb, masks, predicted_boxes):
    df = pd.DataFrame(columns = ['Area, px', 'Confidance, %'])
    total_pores_area_px = 0

    if masks is not None:
        for mask in masks:
            pore_area = mask.cpu().numpy().sum()
            total_pores_area_px += pore_area
            mask = mask.cpu().numpy().reshape(1280, 1280, 1)
            masked_image = np.where(mask.astype(int), np.array([255], dtype='uint8'), mask)
            cimage = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)
            cimage = np.where(cimage.astype(int), np.array([0, 255, 0], dtype='uint8'), np.array([0, 0, 0], dtype='uint8'))
            img_rgb = cv2.addWeighted(img_rgb, 1, cimage, 0.85, 0)
            list_row = [int(pore_area), None]
            df.loc[len(df)] = list_row

        pore_id = 0
        for box, conf in zip(predicted_boxes.xyxy, predicted_boxes.conf):
            cv2.rectangle(img_rgb, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
            label = f"pore#{pore_id}: {str(round(100 * float(conf), 1))}%" #labels[int(box[-1])+1]
            box_label(img_rgb, box, label, (255, 0, 0))
            df.loc[pore_id, "Confidance, %"] = round(100 * float(conf), 1)
            pore_id +=1

    return img_rgb, total_pores_area_px, df


def add_summary(total_pores_area_px, img_rgb, filename):
    blank_pixels = 1280*560
    total_img_area = 1280*1280 - blank_pixels
    pores_percent = total_pores_area_px / total_img_area * 100

    final_img = img_rgb.copy()
    final_img = final_img[250:1100]
    final_img[750:, :] = [0, 0, 0]

    summary = f'Total pores area: {round(pores_percent, 3)} %'
    cv2.putText(final_img, summary, (50, 800), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(final_img, filename, (350, 800), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    #print(total_img_area, total_pores_area_px)
    return final_img, round(pores_percent, 3)


for filename in os.listdir("2H"):
    if filename != ".ipynb_checkpoints":
        IMAGE_PATH = f"2H/{filename}"
        total_pores_area_px = 0
        
        img_rgb, masks, predicted_boxes = sam_yolo(IMAGE_PATH)
        img_rgb, total_pores_area_px, df = draw_bb(img_rgb, masks, predicted_boxes)
        final_img, pores_percent = add_summary(total_pores_area_px, img_rgb, filename)
        print(df)

        list_row = [filename, pores_percent]
        if pores_percent > 0:
            df_porosity.loc[len(df_porosity)] = list_row
        else:
            df_porosity.loc[len(df_porosity)] = [filename, None]

        cv2.imwrite(f"yoloved/yoloved{filename}", final_img)
        #cv2.imshow("final img", final_img)
        

df_porosity.loc['mean'] = ['', round(df_porosity['Porosity, %'].mean(skipna=True), 3)]
df_porosity.to_excel("output.xlsx") 

print(df_porosity)
et = time.time()

elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
