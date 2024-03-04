import os
import tarfile
import io
import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi

proportion_train, proportion_test = 0.8, 0.2

# Extracting data

header_file = "data/ENSSAT_Dataset/header.hdr"
spectral_file = "data/ENSSAT_Dataset/data.pix"
data = envi.open(header_file, spectral_file)
default_red, default_green, default_blue = data.metadata['default bands']
default_red = int(default_red)
default_green = int(default_green)
default_blue = int(default_blue)
nb_bands = data.metadata["bands"]
print(f"Include this in config : {default_red=}, {default_green=}, {default_blue=}, {nb_bands=}")

d = data[:, :, :]

def show_images(images, R=default_red, G=default_green, B=default_blue):
    if type(images) != list:
        images = [images]
    if len(list(images[0].shape)) == 4:
        images = [img for img in images[0]]
    fig, axes = plt.subplots(1, len(images))
    if len(images) == 1:
        axes = [axes]
    for i, img in enumerate(images):
        axes[i].imshow(img[:, :, [R, G, B]])
        axes[i].axis('off')
    plt.show()


# Splitting test/train

train = d[:int(proportion_train * len(d)), :, :]
test = d[-int(proportion_test * len(d)):, :, :]

# Compressing and saving
if not os.path.isdir("data/training"):
    os.makedirs("data/training")
if not os.path.isdir("data/testing"):
    os.makedirs("data/testing")

def save_numpy_array_to_tar_gz(numpy_array, output_tar_gz):
    array_bytes = io.BytesIO()
    np.save(array_bytes, numpy_array)
    array_bytes.seek(0)
    with tarfile.open(output_tar_gz, 'w:gz') as tar:
        info = tarfile.TarInfo(name='numpy_array.npy')
        info.size = len(array_bytes.getvalue())
        tar.addfile(tarinfo=info, fileobj=array_bytes)

save_numpy_array_to_tar_gz(train, "data/training/data.tar.gz")
save_numpy_array_to_tar_gz(test, "data/testing/data.tar.gz")

# Showing 
show_images([d, train, test])

"""
# Open it back just to test
def load_numpy_array_from_tar_gz(tar_gz_file):
    with tarfile.open(tar_gz_file, 'r:gz') as tar:
        numpy_array_file = tar.extractfile('numpy_array.npy')
        numpy_array_bytes = numpy_array_file.read()
        numpy_array = np.load(io.BytesIO(numpy_array_bytes))
    return numpy_array

tar_gz_file = 'data/testing/data.tar.gz'
img = load_numpy_array_from_tar_gz(tar_gz_file)
print(img.shape)

# Second test
import matplotlib.pyplot as plt
R, G, B = 95, 58, 20
X1 = 0
Y1 = 600
X2 = 190
Y2 = 740
plt.imshow(img[X1:X2, Y1:Y2, [R, G, B]])
plt.show()


def mask(source, mask, threshold=0.1):
    filter_result = np.all(mask < threshold, axis=-1)
    source[filter_result] = [0, 0, 0]
    return source

rgb_img = img[:, :, [R, G, B]]
x = mask(rgb_img, img)
plt.imshow(x)
plt.show()
"""
