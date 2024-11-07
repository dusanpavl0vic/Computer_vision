import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2

def f_prime(x):
    return 2 * x


a = 1
fa = f(a)
slope = f_prime(a)


x = np.linspace(-2, 2, 400)
y = f(x)


tangent_x = np.linspace(a - 1, a + 1, 200)
tangent_y = slope * (tangent_x - a) + fa


plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$f(x) = x^2$', color='blue')
plt.plot(tangent_x, tangent_y, label='Tangenta', color='red', linestyle='--')


plt.scatter(a, fa, color='green')  # Tačka (1, 1)
plt.text(a, fa, f'  ({a}, {fa})', fontsize=12, verticalalignment='bottom')


plt.title('Graf funkcije i njena tangenta')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.ylim(-1, 5)
plt.xlim(-2, 2)
plt.show()
