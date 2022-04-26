#!/usr/bin/env python
# coding: utf-8

# In[21]:


"""
I, Andy Le, student number 000805099, certify that all code submitted is my
own work; that I have not copied it from any other source. I also certify that I
have not allowed my work to be copied by others.
"""

import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

numColors = 16
storedKMeans = KMeans()
fromImg = 0

k = np.array([2,4,6,8,10,12,14,16,18,20])
iner = []

plt.figure()
plt.title('OG toonlink')
plt.imshow(toonlink)
plt.axis('off')
plt.show()

for i in k:
    global storedKMeans
    global fromImg
    toonlink = io.imread('toonlink.jpg')
    toonlink = np.array(toonlink, dtype=np.float64) / 255
    w, h, colors = toonlink.shape
    image_data = np.reshape(toonlink, (w * h, colors))
    temp_image_data = shuffle(image_data, random_state=0, n_samples=1000)
    storedKMeans = KMeans(n_clusters=i, random_state=0).fit(temp_image_data)
    l = storedKMeans.predict(image_data)
    temp_image = storedKMeans.cluster_centers_[l].reshape(w, h, -1)
    plt.figure(1)
    plt.clf()
    plt.axis("off")
    plt.title(f"ToonLink quantized ({i} colors)")
    plt.imshow(temp_image)
    print(f"K = {i} SSE = {storedKMeans.inertia_}")
    iner += [(storedKMeans.inertia_)]
        
iner = np.array(iner)

plt.figure(2)    
plt.plot(k,iner)
plt.title("k vs iner")
plt.xlabel("k")
plt.ylabel("iner")
plt.xlim([k.min(), k.max()])
plt.ylim([iner.min(), iner.max()])
plt.show()

global storedKMeans
global fromImg

tengen = io.imread('tengen.jpg')
tengen = np.array(tengen, dtype=np.float64) / 255

w, h, colors = tengen.shape

image_data = np.reshape(tengen, (w * h, colors))
temp_image_data = shuffle(image_data, random_state=0, n_samples=1000)
storedKMeans = KMeans(n_clusters=6, random_state=0).fit(temp_image_data)
l = storedKMeans.predict(image_data)
temp_image = storedKMeans.cluster_centers_[l].reshape(w, h, -1)

plt.figure(3)
plt.clf()
plt.axis("off")
plt.title("OG Tengen")
plt.imshow(tengen)

plt.figure(4)
plt.clf()
plt.axis("off")
plt.title("Tengen quantized")
plt.imshow(temp_image)


# In[ ]:




