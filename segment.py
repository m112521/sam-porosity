import torch
import cv2
import numpy as np

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import matplotlib.pyplot as plt


CHECKPOINT_PATH='sam_vit_h_4b8939.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

IMAGE_PATH = "Test_pore2.png"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam) #SamAutomaticMaskGenerator

model = YOLO('best_s.pt')
results = model.predict(task="detect", source=IMAGE_PATH, conf=0.25)
predicted_boxes = results[0].boxes.xyxy

image_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
# draw boxes
for box in predicted_boxes:
    cv2.rectangle(image_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
cv2.imshow("SAM", image_bgr)
cv2.waitKey(0)

transformed_boxes = mask_predictor.transform.apply_boxes_torch(predicted_boxes, image_bgr.shape[:2])
mask_predictor.set_image(image_bgr)
masks, scores, logits = mask_predictor.predict_torch(boxes = transformed_boxes, multimask_output=False, point_coords=None, point_labels=None)

total_area_px = 0
for mask in masks:
    total_area_px += mask.cpu().numpy().sum()

total_area_prt = 100 * total_area_px / (1280*720) # crop top-bottom white pixels
print(f'Porosity: {round(total_area_prt, 2)}%')


# final_mask = None
# for i in range(len(masks) - 1):
#   if final_mask is None:
#     final_mask = np.bitwise_or(masks[i][0].cpu().numpy(), masks[i+1][0].cpu().numpy())
#   else:
#     final_mask = np.bitwise_or(final_mask, masks[i+1][0].cpu().numpy())

# # visualize the predicted masks
# plt.figure(figsize=(10, 10))
# plt.imshow(image_bgr)
# plt.imshow(final_mask, cmap='gray', alpha=0.7)
# plt.show()