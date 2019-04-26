#!/usr/bin/python
# coding=utf-8

import os
import cv2
import logging
import numpy as np
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

    passer = np.logical_and(vis > 1, nir > 1)
    return np.where(passer, (1.*nir - 1.*vis) / (1.*nir + 1.*vis), 0.)


def calc_rvi(nir, vis):
    """Calculates the RVI of an orthophoto using nir and vis bands.

    :param nir: An array containing the nir band
    :param vis: An array containing the vis band
    :return: An array that will be exported as a tif
    """

    passer = np.logical_and(vis > 1, nir >= 0)
    return np.where(passer, (1. * nir) / (1. * vis), 40.)


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
    return nir_intersection, gre_intersection


def plot_and_save(image, save_name="ndvi.png"):
    plt.figure(figsize=(100, 100))
    # plt.imshow(image, cmap='jet', vmin=-1., vmax=1.)
    plt.imshow(image, cmap='jet', vmin=np.min(image), vmax=np.max(image))
    plt.colorbar(orientation="vertical")

    # plt.hist(np.ravel(image), bins=500)
    plt.savefig(save_name)
    plt.show()
    return


def test():
    base_dir = "/home/pzw/hdd/projects/geo/postprocess/data_in_qingyun/040402"
    gre_transform_dir = os.path.join(base_dir, "uxa040401gre_transparent_mosaic_grayscale.tfw")
    nir_transform_dir = os.path.join(base_dir, "uxa040401nir_transparent_mosaic_grayscale.tfw")

    gre_img_dir = os.path.join(base_dir, "uxa040401gre_transparent_mosaic_grayscale.tif")
    nir_img_dir = os.path.join(base_dir, "uxa040401nir_transparent_mosaic_grayscale.tif")
    nir_img, gre_img = image_intersection(nir_img_dir, gre_img_dir, nir_transform_dir, gre_transform_dir)

    ndvi_img = calc_ndvi(nir_img, gre_img)
    print ndvi_img.shape, "\n", np.max(ndvi_img), np.min(ndvi_img)
    plot_and_save(ndvi_img, "ndvi_qingyun040401r2.png")

    rvi_img = calc_rvi(nir_img, gre_img)
    print rvi_img.shape, "\n", np.max(rvi_img), np.min(rvi_img)
    plot_and_save(rvi_img, "rvi_qingyun040401r0.png")
    return


def main():
    opt = parse()
    nir_image, vis_image = image_intersection(opt.nir, opt.vis, opt.nir_transform, opt.vis_transform)
    if opt.pattern == "rvi":
        rvi_image = calc_rvi(nir_image, vis_image)
        log.info("RVI image max value is {}, min value is {}".format(np.max(rvi_image), np.min(rvi_image)))
        plot_and_save(rvi_image, opt.save_name)
    else:
        ndvi_image = calc_ndvi(nir_image, vis_image)
        log.info("NDVI image max value is {}, min value is {}".format(np.max(ndvi_image), np.min(ndvi_image)))
        plot_and_save(ndvi_image, opt.save_name)
    return


if __name__ == "__main__":
    main()
