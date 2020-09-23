import matplotlib.pyplot as plt
import tensorflow as tf


def plot_image(image):
  fig = plt.gcf()
  fig.set_size_inches(3,3)
  plt.imshow(image, cmap='binary')
  plt.show()


tensor = [
  [[0,0,0], [255,255,255], [255,255,255]],
  [[255,255,255], [0,0,0], [255,255,255]],
  [[255,255,255], [255,255,255], [0,0,0]]
]
plot_image(tensor)