import numpy as np
from canny import *
from tqdm import tqdm

def hough_circle(edge_image, min_radius, max_radius):
    # 初始化Hough累加器
    hough_acc = np.zeros((max_radius - min_radius + 1,) + edge_image.shape)
    # 获取图像中的边缘点
    edge_points = np.where(edge_image > 0) # 返回两个元组，分别是edge点们的y坐标列表和和x坐标列表
    # 对每个边缘点进行处理
    for radius in tqdm(range(min_radius, max_radius + 1), desc='Iterating over radii'):
        for i in range(len(edge_points[0])):
            # 获取边缘点的坐标
            edge_y, edge_x = edge_points[0][i], edge_points[1][i]

            # 计算可能的圆心位置并更新Hough累加器
            for theta in np.arange(0, 360):
                theta = np.deg2rad(theta) # 转换为弧度
                a = edge_x - radius * np.cos(theta) # 计算圆心x坐标
                b = edge_y - radius * np.sin(theta) # 计算圆心y坐标

                if a >= 0 and a < edge_image.shape[1] and b >= 0 and b < edge_image.shape[0]: # 检查圆心是否在图像范围内
                    hough_acc[radius-min_radius, int(b), int(a)] += 1

    return hough_acc


def filter_centers(centers, radii, values, min_distance):
    # 对圆心按照它们在Hough累加器中的值进行排序
    order = np.argsort(values)[::-1] # 步长为-1，即从大到小排序
    centers = centers[order]
    radii = radii[order]

    filtered_centers = []
    filtered_radii = []

    # # 从最大的圆心开始，剔除所有与它距离小于阈值的圆心
    # while len(centers) > 0:
    #     filtered_centers.append(centers[0])
    #     filtered_radii.append(radii[0])
    #     distance = np.sqrt(np.sum((centers - centers[0])**2, axis=1))
    #     # 布尔掩码，标记出距离大于或等于阈值的圆心。
    #     mask = distance >= min_distance
    #     centers = centers[mask]
    #     radii = radii[mask]

    # return np.array(filtered_centers), np.array(filtered_radii)
    return np.array(centers), np.array(radii)


def hough_center_detection(img, min_radius, max_radius, min_distance, num):
    hough_acc = hough_circle(img, min_radius, max_radius)

    # 找到Hough累加器中的所有圆心和半径
    indices = np.arange(hough_acc.size)
    radii, y, x = np.unravel_index(indices, hough_acc.shape)
    radii = radii + min_radius
    centers = np.column_stack((y, x))
    values = hough_acc.flatten()

    # 剔除相邻距离小于阈值的圆心
    filtered_centers, filtered_radii = filter_centers(centers, radii, values, min_distance)

    # 过滤后的前num个圆心坐标和半径
    return filtered_centers[:num], filtered_radii[:num]
    return centers[:num], radii[:num]


