from canny import *
from hough_circle import *
import matplotlib.patches as patches


# 读取图片
img = imageio.imread("31714011104_.pic.jpg")
img = img.mean(axis=-1) # 将图像转换为灰度
print(img.shape)
# canny边缘检测
edge_img = canny(img, sigma=1, lowthresh=0.3, highthresh=0.32)

# hough变换
circle_center, radius = hough_center_detection(edge_img, min_radius=215, max_radius=225, min_distance=10, num=1)
print(f"circle center coodinates: {circle_center}, radius: {radius}")

# 显示结果
plt.imshow(img, cmap='gray')
plt.scatter(circle_center[:,1], circle_center[:,0], color='red') # 标记圆心
for center, radius in zip(circle_center, radius): # 画圆
    circle = patches.Circle((center[1], center[0]), radius, fill=False, color='red', linewidth=2)
    plt.gca().add_patch(circle)
plt.show()