import numpy as np
import cv2
import matplotlib.pyplot as plt


def gamma_correction(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


if __name__ == '__main__':
    img = cv2.imread("buildings.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    hist0, bins0 = np.histogram(img[:, :, 0], bins=np.arange(256))
    hist1, bins1 = np.histogram(img[:, :, 1], bins=np.arange(256))
    hist2, bins2 = np.histogram(img[:, :, 2], bins=np.arange(256))

    #region LAB
    img_new = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    b_hist, bins_b = np.histogram(img_new[:, :, 2], bins=np.arange(256))
    a_hist, bins_a = np.histogram(img_new[:, :, 1], bins=np.arange(256))
    l_hist, bins_l = np.histogram(img_new[:, :, 0], bins=np.arange(256))

    plt.subplot(133)
    plt.bar(bins_b[:-1], b_hist)
    plt.subplot(132)
    plt.bar(bins_a[:-1], a_hist)
    plt.subplot(131)
    plt.bar(bins_l[:-1], l_hist)
    plt.show()

    img_new[:, :, 2] = gamma_correction(img_new[:,:,2], 0.7)

    img_new[:, :, 0] = gamma_correction(img_new[:, :, 0], 0.6)

    new_b_hist, new_bins_b = np.histogram(img_new[:, :, 2], bins=np.arange(256))
    new_l_hist, new_bins_l = np.histogram(img_new[:, :, 0], bins=np.arange(256))


    img_new = cv2.cvtColor(img_new, cv2.COLOR_LAB2RGB)

    plt.subplot(131)
    plt.bar(new_bins_l[:-1], new_l_hist)
    plt.subplot(132)
    plt.bar(bins_a[:-1], a_hist)
    plt.subplot(133)
    plt.bar(new_bins_b[:-1], new_b_hist)
    plt.show()

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_new)
    plt.show()

    #endregion

    #region GAMMAcrveno
    # img_new[:, :, 0] = gamma_correction(img[:, :, 0], 0.7)
    # new_hist, bins_new = np.histogram(img_new[:, :, 0], bins=np.arange(256))

    # plt.subplot(221)
    # plt.bar(x=bins0[:-1], height=hist0)
    # plt.subplot(222)
    # plt.bar(x=bins1[:-1], height=hist1)
    # plt.subplot(223)
    # plt.bar(x=bins2[:-1], height=hist2)
    # plt.subplot(224)
    # plt.bar(x=bins_new[:-1], height=new_hist)
    # plt.show()
    #
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(img_new)
    # plt.show()

    #endregion

    #region EkvalizacijaHistograma
    #img_new[:, :, 1] = cv2.equalizeHist(img_new[:, :, 1]) #zeleni kanal
    #img_new[:, :, 2] = cv2.equalizeHist(img_new[:, :, 2]) #plavi kanal
    # new_hist, bins_new = np.histogram(img_new[:, :, 2], bins=np.arange(256))

    # plt.subplot(221)
    # plt.bar(x=bins0[:-1], height=hist0)
    # plt.subplot(222)
    # plt.bar(x=bins1[:-1], height=hist1)
    # plt.subplot(223)
    # plt.bar(x=bins2[:-1], height=hist2)
    # plt.subplot(224)
    # plt.bar(x=bins_new[:-1], height=new_hist)
    # plt.show()
    #
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(img_new)
    # plt.show()

    #endregion

    #region CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(20, 20))
    # img_new[:, :, 0] = clahe.apply(img[:, :, 0])

    #new_hist, bins_new = np.histogram(img_new[:, :, 0], bins=np.arange(256))

    # plt.subplot(221)
    # plt.bar(x=bins0[:-1], height=hist0)
    # plt.subplot(222)
    # plt.bar(x=bins1[:-1], height=hist1)
    # plt.subplot(223)
    # plt.bar(x=bins2[:-1], height=hist2)
    # plt.subplot(224)
    # plt.bar(x=bins_new[:-1], height=new_hist)
    # plt.show()
    #
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(img_new)
    # plt.show()

    #endregion
