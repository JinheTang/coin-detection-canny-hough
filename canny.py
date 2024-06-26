from scipy import misc
import imageio
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib
plt.rcParams['font.family'] = 'Heiti TC'



# Sobel算子
def sobel_filter(img, direction):
    if direction == 'x':
        Gx = np.array([[-1,0,+1], [-2,0,+2],  [-1,0,+1]])
        result = ndimage.convolve(img, Gx)
    if direction == 'y':
        Gy = np.array([[-1,-2,-1], [0,0,0], [+1,+2,+1]])
        result = ndimage.convolve(img, Gy)
    return result


# 非极大抑制
def non_max_sup(Gmag, Grad):
    NMS = np.zeros(Gmag.shape)
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= -22.5 and Grad[i,j] <= 22.5) or (Grad[i,j] <= -157.5 and Grad[i,j] >= 157.5)):
                if((Gmag[i,j] > Gmag[i,j+1]) and (Gmag[i,j] > Gmag[i,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 22.5 and Grad[i,j] <= 67.5) or (Grad[i,j] <= -112.5 and Grad[i,j] >= -157.5)):
                if((Gmag[i,j] > Gmag[i+1,j+1]) and (Gmag[i,j] > Gmag[i-1,j-1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 67.5 and Grad[i,j] <= 112.5) or (Grad[i,j] <= -67.5 and Grad[i,j] >= -112.5)):
                if((Gmag[i,j] > Gmag[i+1,j]) and (Gmag[i,j] > Gmag[i-1,j])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] >= 112.5 and Grad[i,j] <= 157.5) or (Grad[i,j] <= -22.5 and Grad[i,j] >= -67.5)):
                if((Gmag[i,j] > Gmag[i+1,j-1]) and (Gmag[i,j] > Gmag[i-1,j+1])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0

    return NMS


# 双阈值
def do_thresh_hyst(img, lowThresholdRatio=0.30, highThresholdRatio=0.32):
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio    

    for i in range(1,h-1):
        for j in range(1,w-1):
            if(GSup[i,j] > highThreshold):
                GSup[i,j] = 1
            elif(GSup[i,j] < lowThreshold):
                GSup[i,j] = 0
            else: # 介于高低阈值之间
                if((GSup[i-1,j-1] > highThreshold) or 
                    (GSup[i-1,j] > highThreshold) or
                    (GSup[i-1,j+1] > highThreshold) or
                    (GSup[i,j-1] > highThreshold) or
                    (GSup[i,j+1] > highThreshold) or
                    (GSup[i+1,j-1] > highThreshold) or
                    (GSup[i+1,j] > highThreshold) or
                    (GSup[i+1,j+1] > highThreshold)):
                    GSup[i,j] = 1
    GSup = (GSup == 1) * GSup # 把所有weak edge清零
    return GSup


def canny(img, sigma=1, lowthresh=0.3, highthresh=0.4):
    # 高斯滤波
    img_guassian_filter = ndimage.gaussian_filter(img, sigma)

    # x,y方向sobel滤波
    gx = sobel_filter(img_guassian_filter, 'x') # x方向梯度大小
    gy = sobel_filter(img_guassian_filter, 'y') # y方向梯度大小
    Gmag = np.hypot(gx,gy) # 梯度绝对大小，取sqrt(gx^2+gy^2)
    Grad = np.arctan2(gy,gx) # 梯度方向

    # 非极大抑制
    NMS = non_max_sup(Gmag, Grad)

    # 双阈值
    GSupW = do_thresh_hyst(NMS, lowthresh, highthresh)
    return GSupW



if __name__ == '__main__':
    img = imageio.imread("/Users/bytedance/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/f45079de90e0d29fee35ea5cad92c4b4/Message/MessageTemp/cfcb565991275509e494b88e37240a51/Image/61714011127_.pic.jpg")
    img = img.mean(axis=-1) # 将图像转换为灰度

    # 高斯滤波
    img_guassian_filter = ndimage.gaussian_filter(img, sigma=3)
    plt.imshow(img_guassian_filter, cmap='gray')
    plt.title('高斯滤波模糊化')
    plt.show()

    # sobel滤波
    gx = sobel_filter(img_guassian_filter, 'x') # x方向梯度大小
    gy = sobel_filter(img_guassian_filter, 'y') # y方向梯度大小
    Gmag = np.hypot(gx,gy) # 梯度绝对大小，取sqrt(gx^2+gy^2)
    Grad = np.arctan2(gy,gx) # 梯度方向
    plt.imshow(Gmag, cmap='gray')
    plt.title('sobel滤波提取边界')
    plt.show()

    # 非极大抑制
    NMS = non_max_sup(Gmag, Grad)
    plt.imshow(NMS, cmap='gray')
    plt.title('非极大抑制细化边界')
    plt.show()

    # 双阈值
    GSupW = do_thresh_hyst(NMS, 0.3, 0.4)
    plt.imshow(GSupW, cmap='gray')
    plt.title('双阈值得到最终边界')
    plt.show()
