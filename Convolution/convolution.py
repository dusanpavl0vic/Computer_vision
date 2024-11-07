import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


f = np.array([1, 2, 3, 4, 5])  # Signal
g = np.array([0.2, 0.5, 0.2])   # Kernel


def convolve_1d(signal, kernel):
    kernel = np.flip(kernel)
    signal_length = len(signal)
    kernel_length = len(kernel)
    conv_length = signal_length + kernel_length - 1
    convolved = np.zeros(conv_length)
    
    for i in range(conv_length):
        for j in range(kernel_length):
            if 0 <= i - j < signal_length:
                convolved[i] += signal[i - j] * kernel[j]
    return convolved

result = convolve_1d(f, g)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.stem(f)
plt.title('Signal f')
plt.subplot(1, 3, 2)
plt.stem(g)
plt.title('Signal g')
plt.subplot(1, 3, 3)
plt.stem(result)
plt.title('Konvolucija f * g')
plt.show()


image = np.array([[0, 1, 2, 1, 0],
                  [1, 2, 3, 2, 1],
                  [2, 3, 4, 3, 2],
                  [1, 2, 3, 2, 1],
                  [0, 1, 2, 1, 0]])

kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]]) / 16

result_image = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Izvorna slika')
plt.subplot(1, 2, 2)
plt.imshow(result_image, cmap='gray')
plt.title('Slika nakon konvolucije')
plt.show()
