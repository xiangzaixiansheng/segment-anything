from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    



# sam_checkpoint = "sam_vit_h_4b8939.pth"
sam_checkpoint = "/Users/hanxiang1/work/a-learn/segment-anything/model/sam_vit_h_4b8939.pth"
model_type = "vit_h"

# device = "cuda"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

image = cv2.imread("/Users/hanxiang1/work/a-learn/segment-anything/notebooks/images/truck.jpg")
# None 表示没有指定新的尺寸，fx 和 fy 参数来指定缩放因子 fx=0.5 和 fy=0.5 表示图片的宽度和高度都被缩放为原来的一半
image = cv2.resize(image,None,fx=0.5,fy=0.5)
#  函数用于将图片从一个颜色空间转换到另一个。这里它将图片从 BGR 格式转换为 RGB 格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


predictor = SamPredictor(sam)
predictor.set_image(image)


input_point = np.array([[250, 187]])
input_label = np.array([1])

#=============
# 单点
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# print(plt.gca())
# plt.axis('on')
# plt.show()  

##=============
# 单点多模式  multimask_output=True
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print(masks.shape)  # (number_of_masks) x H x W  | output (3, 600, 900)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

##=============
# # 多点模式
# input_point = np.array([[250, 184], [562, 322]])
# input_label = np.array([1, 1])

# # 需要前面的预测的logits
# mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

# masks, _, _ = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     mask_input=mask_input[None, :, :],
#     multimask_output=False,
# )
# print(masks.shape) #output: (1, 600, 900)

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_mask(masks, plt.gca())
# show_points(input_point, input_label, plt.gca())
# plt.axis('off')
# plt.show() 


#=============
# boxes输入生成mask
# input_box = np.array([212, 300, 350, 437])
# masks, _, _ = predictor.predict(
#     point_coords=None,
#     point_labels=None,
#     box=input_box[None, :],
#     multimask_output=False,
# )
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# show_mask(masks[0], plt.gca())
# show_box(input_box, plt.gca())
# plt.axis('off')
# plt.show()

