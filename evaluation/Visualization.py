import cv2
import numpy as np
import os
from utils import utils_image

class PartialEnlargement:
    @staticmethod
    def partial_enlarge_one_img(image_path, region_positions, region_width, region_height):
        """
        在图像上用红框标注特定区域，并在原图下方拼接区域

        Parameters
        ----------
        image_path : 图像文件路径
        region_positions : 区域位置序列
        region_width : 区域宽度
        region_height : 区域高度

        Returns
        -------

        """

        # 读取图像
        img = cv2.imread(image_path)
        (h, w, c) = img.shape[:]

        region_num = len(region_positions)
        factor = int(w/region_num) / region_width
        new_h = h + int (region_height * factor)

        # 创建白色背景的新图像
        new_image = np.ones((new_h, w, c), dtype=np.uint8) * 255

        # 将原图放置在上方
        new_image[0:h, 0:w] = img

        # 提取每个区域
        for i, position in enumerate(region_positions):
            x, y = position[:]

            # 绘制红色矩形框标注感兴趣区域
            cv2.rectangle(new_image, (x, y), (x+region_width, y+region_height), (0, 0, 255), 2)

            # 提取感兴趣区域
            roi = img[y:y+region_height, x:x+region_width]

            # 放大区域
            roi_zoomed = cv2.resize(roi, None, fx=factor,fy=factor,interpolation=cv2.INTER_LINEAR)
            (zoom_h, zoom_w) = roi_zoomed.shape[:2]

            # 将放大区域叠加到原图上
            new_image[h : h+zoom_h, i*zoom_w : (i+1)*zoom_w] = roi_zoomed

            # 在放大区域周围绘制边框
            cv2.rectangle(new_image, (i*zoom_w, h), ((i+1)*zoom_w, h+zoom_h), (0, 0, 255), 1)

        return new_image

    @staticmethod
    def partial_enlarge_all_img(img_dir, region_positions, region_width, region_height, output_dir="output"):
        '''
        批量标注和拼接区域

        Parameters
        ----------
        img_dir : 图像目录
        region_positions : 区域位置序列
        region_width : 区域宽度
        region_height : 区域高度
        output_dir : 保存目录

        Returns
        -------

        '''

        img_paths, names = utils_image.FileHandler.list_img_paths(img_dir)

        # 创建输出目录
        utils_image.FileHandler.make_dir(output_dir)

        results = []


        for i, img_path in enumerate(img_paths):
            result_img = PartialEnlargement.partial_enlarge_one_img(img_path, region_positions, region_width, region_height)
            results.append(result_img)
            name = names[i]+'.png'
            # 保存结果
            output_path = os.path.join(output_dir, name)
            cv2.imwrite(output_path, result_img)

        return results

# 使用示例
if __name__ == "__main__":
    img_dir = os.path.join(os.getcwd(), 'test')
    output_dir = os.path.join(os.getcwd(), 'output')
    # 定义区域列表，左上点坐标(x,y)
    regions = [
        (140, 280),    # 区域1
        (300, 280),  # 区域2
    ]
    region_width = 80
    region_height = 50

    # result = annotate_and_zoom(image_path, regions, region_width,region_height)
    result = PartialEnlargement.partial_enlarge_all_img(img_dir, regions, region_width, region_height, output_dir=output_dir)

