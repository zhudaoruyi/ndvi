#!/usr/bin/python
# coding=utf-8

import os
import cv2
import logging
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)
log = logging.getLogger()


def parse():
    parser = ArgumentParser()
    parser.add_argument('-n', '--nir', required=True,
                        help='NIR band image directory')
    parser.add_argument('-v', '--vis', required=True,
                        help='vis band image directory')
    parser.add_argument('-nt', '--nir_transform', required=True,
                        help='NIR transform array file directory')
    parser.add_argument('-vt', '--vis_transform', required=True,
                        help='vis transform array file directory')
    parser.add_argument('-p', '--pattern', choices=["ndvi", "rvi"], default="ndvi",
                        help='calculate ndvi or rvi')
    parser.add_argument('-s', '--save_name', default="ndvi.png",
                        help='ndvi image save name')
    return parser.parse_args()


def write_tif(filename, im_data, im_geotrans=None, im_proj=None):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape  # channel first np.transpose(x, (2, 0, 1))
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    if im_geotrans is not None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    if im_proj is not None:
        dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])  # b g r
    del dataset
    return


def pixel2geo(geomatrix, col, row):
    """pixel to geo coordination

    :param geomatrix: geo transform
    :param col: pixel column
    :param row: pixel row
    :return: geo coordination
    """
    X = geomatrix[4] + geomatrix[0] * col + geomatrix[2] * row
    Y = geomatrix[5] + geomatrix[1] * col + geomatrix[3] * row

    # Shift to the center of the pixel
    X += geomatrix[0] / 2.0
    Y += geomatrix[3] / 2.0
    return X, Y


def geo2pixel(geomatrix, Xg, Yg):
    """geo coordination to pixel

    :param geomatrix: geo transform
    :param Xg: longitude
    :param Yg: latitude
    :return: pixel column, pixel row
    """
    g = geomatrix
    col = (g[2] * Yg - g[3] * Xg + g[3] * g[4] - g[2] * g[5] + g[3] * (g[0] - g[2]) / 2.) / (g[1] * g[2] - g[0] * g[3])
    row = (g[0] * Yg - g[1] * Xg + g[1] * g[4] - g[0] * g[5] + g[0] * (g[1] - g[3]) / 2.) / (g[0] * g[3] - g[1] * g[2])
    return int(round(col)), int(round(row))


def geo_intersection(vis_tr, nir_tr, vis_shape, nir_shape):
    """geo intersection of vis band and nir band

    :param vis_tr: vis band transform array
    :param nir_tr: NIR band transform array
    :param vis_shape: vis band image shape
    :param nir_shape: NIR band image shape
    :return: ulx -> up left x, uly -> up left y, drx -> down right x, dry -> down right y
    """
    ulx = max(vis_tr[4], nir_tr[4])
    uly = min(vis_tr[5], nir_tr[5])

    x_vis, y_vis = pixel2geo(vis_tr, vis_shape[1], vis_shape[0])
    x_nir, y_nir = pixel2geo(nir_tr, nir_shape[1], nir_shape[0])
    drx = min(x_vis, x_nir)
    dry = max(y_vis, y_nir)
    return ulx, uly, drx, dry


def calc_ndvi(nir, vis):
    """Calculates the NDVI of an orthophoto using nir and vis bands.

    :param nir: An array containing the nir band
    :param vis: An array containing the vis band
    :return: An array that will be exported as a tif
    """

    # mask = np.not_equal((nir + vis), 0.0)
    # # return np.choose(mask, (-1., np.true_divide(np.subtract(nir, vis), np.add(nir, vis))))
    # return np.choose(mask, (0., (1.*nir - 1.*vis) / (1.*nir + 1.*vis)))

    # passer = np.logical_and(vis > 1, nir > 1)
    passer = np.logical_and(vis > 0, nir > 0)
    # return np.where(passer, (1.*nir - 1.*vis) / (1.*nir + 1.*vis), 0.)
    # return np.where(passer, (1.*nir - 1.*vis) / (1.*nir + 1.*vis), -np.inf)  # 负无穷代表边缘无地块区域
    return np.where(passer, (1.*nir - 1.*vis) / (1.*nir + 1.*vis), -1e388)  # arcgis


def calc_rvi(nir, vis):
    """Calculates the RVI of an orthophoto using nir and vis bands.

    :param nir: An array containing the nir band
    :param vis: An array containing the vis band
    :return: An array that will be exported as a tif
    """

    passer = np.logical_and(nir > 0, vis > 0)
    return np.where(passer, (1. * nir) / (1. * vis), -1)  # 周边无像素的黑边非地块区域设置为-1
    # return np.where(passer, (1. * nir) / (1. * vis), -np.inf)  # 负无穷代表边缘无地块区域
    # return np.where(passer, (1. * nir) / (1. * vis), -1e388)  # arcgis


def image_intersection(nir_dir, vis_dir, nir_transform_dir, vis_transform_dir):
    """image intersection of vis band and nir band

    :param nir_dir: NIR image directory
    :param vis_dir: vis image(like GREEN band) directory
    :param nir_transform_dir: NIR image transform array file directory
    :param vis_transform_dir: vis image transform array file directory
    :return:
    """

    with open(vis_transform_dir) as fr:
        gre_transform = fr.read()
    gre_transform = [float(value) for value in gre_transform.splitlines()]
    log.info("vis band transform array is {}".format(gre_transform))

    with open(nir_transform_dir) as fr:
        nir_transform = fr.read()
    nir_transform = [float(value) for value in nir_transform.splitlines()]
    log.info("NIR band transform array is {}".format(nir_transform))

    gre_img = cv2.imread(vis_dir, 0)  # grey
    log.info("vis band image shape is {}".format(gre_img.shape[:2]))
    nir_img = cv2.imread(nir_dir, 0)  # grey
    log.info("NIR band image shape is {}".format(nir_img.shape[:2]))

    ulx, uly, drx, dry = geo_intersection(gre_transform, nir_transform, gre_img.shape, nir_img.shape)

    col_l_gre, row_l_gre = geo2pixel(gre_transform, ulx, uly)
    col_r_gre, row_r_gre = geo2pixel(gre_transform, drx, dry)
    log.info("vis band image pixel range {}\t{}".format((col_l_gre, row_l_gre), (col_r_gre, row_r_gre)))

    col_l_nir, row_l_nir = geo2pixel(nir_transform, ulx, uly)
    col_r_nir, row_r_nir = geo2pixel(nir_transform, drx, dry)
    log.info("NIR band image pixel range {}\t{}".format((col_l_nir, row_l_nir), (col_r_nir, row_r_nir)))

    gre_intersection = gre_img[max(0, row_l_gre):min(gre_img.shape[0], row_r_gre),
                       max(0, col_l_gre):min(gre_img.shape[1], col_r_gre)]
    nir_intersection = nir_img[max(0, row_l_nir):min(nir_img.shape[0], row_r_nir),
                       max(0, col_l_nir):min(nir_img.shape[1], col_r_nir)]
    log.info("vis image intersection shape is {}".format(gre_intersection.shape))
    log.info("NIR image intersection shape is {}".format(nir_intersection.shape))

    # keep the same image size
    resize_ratio = gre_transform[0] / nir_transform[0]
    # if nir_transform[0] > gre_transform[0]:  # 则nir shape < gre shape，GSD(m/pixel)越大，同一个地块(地理坐标一样)，分辨率(尺寸)就越小
    gre_intersection = cv2.resize(gre_intersection,
                                  (int(gre_intersection.shape[1] * resize_ratio), int(gre_intersection.shape[0] * resize_ratio)))
    log.info("vis image intersection shape is {}".format(gre_intersection.shape))
    log.info("NIR image intersection shape is {}".format(nir_intersection.shape))

    gre_intersection = gre_intersection[:min(gre_intersection.shape[0], nir_intersection.shape[0]),
                       :min(gre_intersection.shape[1], nir_intersection.shape[1])]
    nir_intersection = nir_intersection[:min(gre_intersection.shape[0], nir_intersection.shape[0]),
                       :min(gre_intersection.shape[1], nir_intersection.shape[1])]

    assert gre_intersection.shape == nir_intersection.shape

    log.info("intersection image shape is {}".format(nir_intersection.shape))

    # with open(os.path.basename(nir_transform_dir), "w") as fw:
    #     fw.write("\n".join([str(i) for i in [nir_transform[0], 0, 0, nir_transform[3], ulx, uly]]))

    # return nir_intersection, gre_intersection
    # return nir_intersection, gre_intersection, [nir_transform[0], 0, 0, nir_transform[3], ulx, uly]  # pix4d
    return nir_intersection, gre_intersection, [ulx, nir_transform[0], 0, uly, 0, nir_transform[3]]  # arcgis


def plot_and_save(image, save_name="ndvi.png", pattern="ndvi"):
    # plt.figure(figsize=(int(image.shape[1]/100), int(image.shape[0]/100)), dpi=10)  # figsize * dpi 即为图像输出尺寸
    plt.figure(figsize=(image.shape[1], image.shape[0]), dpi=1)  # figsize * dpi 即为图像输出尺寸
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    if pattern == "rvi":
        # plt.imshow(image, cmap='jet', vmin=0., vmax=1.)
        # plt.imshow(image, cmap='jet_r', vmin=0., vmax=1.)
        # plt.imshow(image, cmap='seismic', vmin=0., vmax=1.)
        plt.imshow(image, cmap='RdYlGn', vmin=0., vmax=1.)
        # plt.imshow(image, cmap='RdYlBu', vmin=0., vmax=1.)
        # plt.imshow(image, cmap='PiYG', vmin=0., vmax=1.)
        # plt.imshow(image, cmap='Spectral', vmin=0., vmax=1.)
        # plt.imshow(image, cmap='hsv_r', vmin=0., vmax=1.)
    else:
        plt.imshow(image, cmap='jet', vmin=-1., vmax=1.)

    # plt.imshow(image, cmap='jet', vmin=np.min(image), vmax=np.max(image))
    # plt.colorbar(orientation="vertical")  # 颜色轴（图例）
    plt.axis("off")

    # image[image == -np.inf] = 0.4
    # plt.hist(np.ravel(image), bins=50)
    # plt.xlim([0.5, 1.0])
    plt.savefig(save_name)
    plt.show()
    return


# def plot_and_save2(image, save_name="ndvi.png"):
#     fig, ax = plt.subplots(figsize=(50, 50), dpi=256, frameon=False, squeeze=True)
#     # ndvi_map = ax.imshow(image, cmap='seismic', vmin=-1, vmax=1)  # colormap = RdYlGn seismic PiYG jet
#     ndvi_map = ax.imshow(image, cmap='jet', vmin=np.min(image), vmax=np.max(image))
#     # fig.colorbar(ndvi_map, fraction=.05)
#     # ax.set(title="test")
#     ax.set_axis_off()
#     plt.show()
#     return


# def test():
#     base_dir = "/home/pzw/hdd/projects/geo/postprocess/data_in_qingyun/040402"
#     gre_transform_dir = os.path.join(base_dir, "uxa040401gre_transparent_mosaic_grayscale.tfw")
#     nir_transform_dir = os.path.join(base_dir, "uxa040401nir_transparent_mosaic_grayscale.tfw")
#
#     gre_img_dir = os.path.join(base_dir, "uxa040401gre_transparent_mosaic_grayscale.tif")
#     nir_img_dir = os.path.join(base_dir, "uxa040401nir_transparent_mosaic_grayscale.tif")
#     nir_img, gre_img = image_intersection(nir_img_dir, gre_img_dir, nir_transform_dir, gre_transform_dir)
#
#     ndvi_img = calc_ndvi(nir_img, gre_img)
#     print ndvi_img.shape, "\n", np.max(ndvi_img), np.min(ndvi_img)
#     plot_and_save(ndvi_img, "ndvi_qingyun040401r2.png")
#
#     rvi_img = calc_rvi(nir_img, gre_img)
#     print rvi_img.shape, "\n", np.max(rvi_img), np.min(rvi_img)
#     plot_and_save(rvi_img, "rvi_qingyun040401r0.png")
#     return


def postprocess(img, th=0.6):
    """参照色标卡，生成热力图

    :param img: RVI or NDVI image
    :return: heatmap image
    """
    # 色标卡
    colormap = cv2.imread("images/colormap02.jpg")
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)[0]  # 读取色标卡的一个列
    log.info("colormap shape is {}".format(colormap.shape))

    a = 1. / (1. - th)
    b = th / (th - 1.)
    src_img = list()
    for i in range(img.shape[0]):
        row = list()
        for j in range(img.shape[1]):
            if img[i][j] < 0:  # 周边黑色的非地块区域全部设置为白色
                row.append(np.array([255, 255, 255], np.uint8))
            elif img[i][j] < th:  # 根据整个地块RVI灰度直方图，可以看出值集中分布在0.5～1之间
                row.append(colormap[int(10 * img[j][j])])
            else:
                row.append(colormap[int((colormap.shape[0] - 1) * (img[i][j] * a + b))])
        src_img.append(np.array(row))
    src_img = np.array(src_img)
    log.info("output image shape is {}".format(src_img.shape))
    return src_img


def main():
    opt = parse()
    nir_image, vis_image, im_geo = image_intersection(opt.nir, opt.vis, opt.nir_transform, opt.vis_transform)
    if opt.pattern == "rvi":
        rvi_image = calc_rvi(nir_image, vis_image)
        log.info("RVI image max value is {}, min value is {}".format(np.max(rvi_image), np.min(rvi_image)))
        rvi_image[rvi_image > 1] = 1
        # write_tif(opt.save_name+".tif", rvi_image * 255, im_geo)  # 单波段灰图
        post_img = postprocess(rvi_image)
        # cv2.imwrite(opt.save_name, post_img)
        cv2.imwrite(opt.save_name, cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB))
        write_tif(opt.save_name+".tif", np.transpose(post_img, (2, 0, 1)), im_geo)
        # plot_and_save(rvi_image, opt.save_name, opt.pattern)
    else:
        ndvi_image = calc_ndvi(nir_image, vis_image)
        log.info("NDVI image max value is {}, min value is {}".format(np.max(ndvi_image), np.min(ndvi_image)))
        plot_and_save(ndvi_image, opt.save_name, opt.pattern)
    return


if __name__ == "__main__":
    main()
