import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
from scipy.ndimage import binary_dilation, binary_erosion

# 识别填充颜色
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# 识别添加颜色 同时 边缘增加边框
def show_anns_colored_edges(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0  # 设置透明背景

    for ann in sorted_anns:
        m = ann['segmentation']

        # 创建边缘掩膜
        dilated_mask = binary_dilation(m)
        eroded_mask = binary_erosion(m)
        edge_mask = dilated_mask & ~eroded_mask

        # 应用颜色掩膜
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # 随机颜色+不透明度
        img[m] = color_mask

        # 应用蓝色边缘
        blue_edge_color = np.array([0, 0, 1, 1])  # 蓝色+不透明
        img[edge_mask] = blue_edge_color
    ax.imshow(img)


# 保存切图
def extract_and_save(image, mask, filename):
    # 确保掩膜是布尔类型
    bool_mask = mask.astype(bool)


    # 计算掩膜覆盖的区域大小
    mask_area = np.sum(bool_mask)
    min_area = 2000
    print(f" mask_area{mask_area} filaname {filename}")

    # 如果掩膜区域小于最小面积，则不保存
    if mask_area < min_area:
        print(f"Skipping small mask with area {mask_area}")
        return

    # 创建一个空白图像，用于存放结果
    extracted_image = np.zeros_like(image)

    # 提取每个通道
    for i in range(3):  # 对于RGB的每个通道
        extracted_image[:,:,i] = image[:,:,i] * bool_mask

    # 保存提取的图像
    cv2.imwrite(filename, cv2.cvtColor(extracted_image, cv2.COLOR_RGB2BGR))

image = cv2.imread('input/7e6ada89edef4747813292ef383e7710.png')
image = cv2.resize(image,None,fx=0.5,fy=0.5)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# sam_checkpoint = "sam_vit_h_4b8939.pth"
# 模型地址
sam_checkpoint = "/Users/hanxiang1/work/a-learn/segment-anything/model/sam_vit_h_4b8939.pth"
model_type = "vit_h"

# device = "cuda"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


# 手动调参数
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=50,#控制采样点的间隔，值越小，采样点越密集 10 32
    pred_iou_thresh=0.86,#mask的iou阈值
    stability_score_thresh=0.92,#mask的稳定性阈值
    crop_n_layers=1, # 这个参数可能控制在生成掩膜时考虑的图像层数。不同的层数可能会影响掩膜的细节和复杂度
    crop_n_points_downscale_factor=2, # 这可能是在生成掩膜时对采样点进行降采样的因子。更高的值可能会减少用于生成掩膜的点的数量，从而影响掩膜的精确度
    min_mask_region_area=40,  #最小mask面积，会使用opencv滤除掉小面积的区域 50
)
masks2 = mask_generator_2.generate(image)

print(len(masks2)) # 69

plt.figure(figsize=(10,10))
plt.imshow(image)

# 增加mask u阿妈色
#show_anns(masks2)

# 增加mask颜色和边框
show_anns_colored_edges(masks2)
plt.axis('off')

# plt.show() 

# 保存当前图表
plt.savefig(f"output/auto1/mask_auto.png")
plt.close()

# 迭代处理每个掩膜并保存
for i, mask in enumerate(masks2):
    mask = mask['segmentation']
    extract_and_save(image, mask, f"output/auto1/extracted_{i+1}.png")