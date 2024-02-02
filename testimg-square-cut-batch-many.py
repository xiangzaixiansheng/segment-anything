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


# 裁剪区域 masks 数组
def save_and_crop_with_transparency(image, masks, filename):
    # 创建一个与原始图像相同大小的透明背景图像
    cropped_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    # 叠加所有掩膜
    for mask in masks:
         # 如果掩膜是 PyTorch 张量，将其转换为 NumPy 数组
        if isinstance(mask, torch.Tensor):
            # 确保掩膜在 CPU 上，然后转换为 NumPy 数组
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        bool_mask = mask_np.astype(bool)

        # 对于每个掩膜，复制颜色通道
        for i in range(3):  # 对于RGB的每个通道
            cropped_image[:,:,i] = np.where(bool_mask, image[:,:,i], cropped_image[:,:,i])
        # 更新透明通道
        cropped_image[:,:,3] = np.where(bool_mask, 255, cropped_image[:,:,3])  # 255 表示不透明

    # 保存带有透明背景的合并掩膜图像
    cv2.imwrite(filename, cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGRA))


# sam_checkpoint = "sam_vit_h_4b8939.pth"
sam_checkpoint = "/Users/hanxiang1/work/a-learn/segment-anything/model/sam_vit_h_4b8939.pth"
model_type = "vit_h"

# device = "cuda"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

img_path = "input/136da2f59f3b4729a7ebd158db2fa713.png"
image = cv2.imread(img_path)
# None 表示没有指定新的尺寸，fx 和 fy 参数来指定缩放因子 fx=0.5 和 fy=0.5 表示图片的宽度和高度都被缩放为原来的一半
# image = cv2.resize(image,None,fx=0.5,fy=0.5)
#  函数用于将图片从一个颜色空间转换到另一个。这里它将图片从 BGR 格式转换为 RGB 格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


predictor = SamPredictor(sam)
predictor.set_image(image)


#=============
# boxes输入生成mask 多次的
# (x_min, y_min, x_max, y_max) (x_min, y_min) 是框左上角的坐标，(x_max, y_max) 是框右下角的坐标
# input_box = np.array([408, 762, 735, 1026], [433, 911, 556, 998])


transformed_coords = None
input_label = None
input_point = None

# 多box传输
boxes = np.array([ [408, 762, 735, 1026],[401, 1072, 716, 1320]])
input_boxes = torch.tensor(boxes, device=predictor.device)
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, scores, _ = predictor.predict_torch(
    point_coords=transformed_coords,
    point_labels=input_label,
    boxes=transformed_boxes,
    multimask_output=True,
)

print(masks.shape)  # (batch_size) x (num_predicted_masks_per_input) x H x W | output: torch.Size([4, 1, 600, 900])

print("masks length", len(masks))
print("=======================")

for i, (mask_item_arry, score) in enumerate(zip(masks, scores)): 
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    print(mask_item_arry)
    print("leng mask_item_arry", len(mask_item_arry))
    print("leng masks[i]", len(masks[i]))

    for mask_item in mask_item_arry:
        show_mask(mask_item.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    # plt.title(f"Square Mask Many {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f"square_mask_many_{i+1}.png")
    plt.close()
    # 裁剪并保存掩膜区域
    save_and_crop_with_transparency(image, masks, f"cropped_square_mask_many_{i+1}.png")



# 这种形式不可用 多个box 不能批量生成