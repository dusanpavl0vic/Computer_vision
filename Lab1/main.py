import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter

def fft(img):
    img_fft = np.fft.fft2(img)
    img_fft_shift = np.fft.fftshift(img_fft)
    return img_fft_shift


def inverse_fft(magnitude, phase):

    img_fft = magnitude * np.exp(1j * phase)
    img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft)))

    return img_filtered

def fft_noise_removed(img):

    img_fft = fft(img)

    img_fft[156, 236] = 0
    img_fft[156, 276] = 0
    img_fft[356, 236] = 0
    img_fft[356, 276] = 0

    img_fft_mag = np.abs(img_fft)
    img_fft_log = np.log(1 + img_fft_mag)
    img_fft_phase = np.angle(img_fft)

    plt.title("Amplituda bez suma")
    plt.imshow(img_fft_log, cmap="gray")
    cv2.imwrite("fft_outnoise.png", img_fft_log)
    plt.show()

    img_filtered = inverse_fft(img_fft_mag, img_fft_phase)

    return img_filtered



def apply_gaussian_filter(img, kernel_size=5):
    
    img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img_blur

def apply_median_filter(img, size=3):
    
    img_filtered = median_filter(img, size)
    return img_filtered



def low_pass_filter(img, center):

    radius = 100
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_fft_log = np.log(1 + img_fft_mag)
    img_fft_phase = np.angle(img_fft)



    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) > radius*radius: # jednacina kruga, zelimo da sve izvan kruga bude nula sto predstavlja idealni low pass filter
                img_fft_log[x,y] = 0

    img_fft_mag = np.exp(img_fft_log) - 1


    plt.imshow(img_fft_log, cmap="gray")
    plt.show()

    img_filtered = inverse_fft(img_fft_mag, img_fft_phase)

    return img_filtered


if __name__ == "__main__":
    img = cv2.imread("slika_0.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(img, cmap="gray")
    plt.title("Slika sa sumom")
    #cv2.imwrite("input.png", img)
    plt.show()

    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)
    img_fft_log = np.log(1 + img_fft_mag)
    img_fft_phase = np.angle(img_fft)

    plt.imshow(img_fft_log, cmap="gray")
    plt.title("Amplituda sa susmom")
    cv2.imwrite("fft_noise.png", img_fft_log)
    plt.show()
    
    img_filtered = fft_noise_removed(img)

    plt.imshow(img_filtered, cmap="gray")
    plt.title("Slika bez suma")
    cv2.imwrite("img_outnoise.png", img_filtered)
    plt.show()

    #Pokusaj pomocu gausovog filtera
    img_filtered = apply_gaussian_filter(img, kernel_size=11)

    plt.imshow(img_filtered, cmap="gray")
    plt.title("Slika nakon Gaussovog filtriranja")
    plt.show()

    img_filtered = apply_median_filter(img, size=8)

    plt.imshow(img_filtered, cmap="gray")
    plt.title("Slika nakon Median filtriranja")
    plt.show()


    center = (256, 256)
    img_filtered = low_pass_filter(img, center)

    plt.imshow(img_filtered, cmap="gray")
    plt.title("Slika nakon otklanjanja visokih frekvencija")
    plt.show()

