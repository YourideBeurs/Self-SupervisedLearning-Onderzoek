import numpy as np
from glob import glob
from pydicom import dcmread
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython import display

from glob import glob
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


def gpu_check():
    return len(tf.config.list_physical_devices('GPU')) == 2


def limit_gpu_memory(memory_limt: int):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limt)])
            tf.config.experimental.set_virtual_device_configuration(
                gpus[1], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limt)])
        except RuntimeError as e:
            print(e)


def read_dicom_slices(path: str) -> np.array:
    slice_files = np.array([dcmread(p) for p in glob(f'{path}/*')])

    sorted_slice_files = slice_files[
        np.argsort(  # Sort the slices according to the vertical position in the slice array
            [float(s.ImagePositionPatient[-1]) for s in slice_files]  # ...listed in the ImagePositionPatient metadata
        )]
    return np.rollaxis(axis=-1,  # rollaxis to ensure the first axis are is the slice index
                       a=np.dstack([s.pixel_array for s in sorted_slice_files]))


def display_dicom_slice(scan: np.array):
    plt.imshow(scan, cmap=plt.cm.gray)


def display_dicom_slices(scan_array: np.array):
    fig = plt.figure(figsize=(40, 40))

    index = 1
    for image in scan_array:
        f = plt.subplot(10, 8, index)
        plt.imshow(image, cmap=plt.cm.gray)
        index += 1

    plt.show()


def _generate_random_xy_topleft(image_shape: int, patch_size: int) -> tuple:
    topleft_range_x = image_shape[0] - patch_size
    topleft_range_y = image_shape[1] - patch_size

    patch_topleft_x = np.random.randint(topleft_range_x)
    patch_topleft_y = np.random.randint(topleft_range_y)

    return (patch_topleft_x, patch_topleft_y)


def _is_overlap(patch1: tuple, patch2: tuple, patch_size: int) -> bool:
    x1min = patch1[0]
    x1max = patch1[0] + patch_size
    y1min = patch1[1]
    y1max = patch1[1] + patch_size
    x2min = patch2[0]
    x2max = patch2[0] + patch_size
    y2min = patch2[1]
    y2max = patch2[1] + patch_size

    return ((x1min < x2max) and (x2min < x1max) and (y1min < y2max) and (y2min < y1max))


def _swap_patches(image: np.array, patch1: tuple, patch2: tuple, patch_size: int):
    temp_buffer = np.empty((patch_size, patch_size))
    temp_buffer = image[patch1[0]:patch1[0] + patch_size, patch1[1]:patch1[1] + patch_size].copy()  
    
    image[patch1[0]:patch1[0] + patch_size, patch1[1]:patch1[1] + patch_size] = image[patch2[0]:patch2[0] + patch_size,
                                                                                patch2[1]:patch2[1] + patch_size]
    image[patch2[0]:patch2[0] + patch_size, patch2[1]:patch2[1] + patch_size] = temp_buffer


def corrupt_image(image: np.array, num_switches: int, patch_size: int) -> np.array:
    corrupted_image = image.copy()

    for i in range(num_switches):
        while True:
            patch1 = _generate_random_xy_topleft(corrupted_image.shape, patch_size)
            patch2 = _generate_random_xy_topleft(corrupted_image.shape, patch_size)
            if not _is_overlap(patch1, patch2, patch_size):
                break
        _swap_patches(corrupted_image, patch1, patch2, patch_size)

    return corrupted_image


def corrupt_3d_image(image_3d: np.array, num_switches: int, patch_size: int) -> np.array:
    corrupted_image_3d = image_3d.copy()

    for i in range(num_switches):
        image_first_slice = corrupted_image_3d[0]
        while True:
            patch1 = _generate_random_xy_topleft(image_first_slice.shape, patch_size)
            patch2 = _generate_random_xy_topleft(image_first_slice.shape, patch_size)
            if not _is_overlap(patch1, patch2, patch_size):
                break
        
        random_slice_number_first = np.random.randint(0, image_3d.shape[0]-1)
        random_slice_number_second = np.random.randint(random_slice_number_first+1, image_3d.shape[0])

        for image in corrupted_image_3d[random_slice_number_first:random_slice_number_second+1]:
            _swap_patches(image, patch1, patch2, patch_size)

    return corrupted_image_3d


def create_2d_max(scan_array: np.array) -> np.array:
    return np.max(scan_array, axis=0)


def create_2d_sum(scan_array: np.array) -> np.array:
    return np.sum(scan_array, axis=0)


def display_triplets(image: np.array, image_corrupt: np.array, image_predict: np.array):
    fig = plt.figure(figsize=(10, 30))

    f = plt.subplot(1, 3, 1)
    plt.title('original')
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray)
    f = plt.subplot(1, 3, 2)
    plt.title('corrupted')
    plt.axis('off')
    plt.imshow(image_corrupt, cmap=plt.cm.gray)
    f = plt.subplot(1, 3, 3)
    plt.title('predicted')
    plt.axis('off')
    plt.imshow(image_predict, cmap=plt.cm.gray)
    
    plt.show()


def display_3d_scan(scan: np.array):
    Figure = plt.figure()
    
    Figure.set_size_inches(4, 4, True)
    
    plt.imshow(scan[0], cmap=plt.cm.gray)
    plt.axis('off')
    
    def AnimationFunction(frame): 
        plt.imshow(scan[frame], cmap=plt.cm.gray)
    
    anim = FuncAnimation(Figure, AnimationFunction, frames=scan.shape[0], interval=100)
    
    html = display.HTML(anim.to_jshtml())
    display.display(html)
    
    plt.close()
    
    
def display_3d_triplets(scan: np.array, scan_corrupt: np.array, scan_predict: np.array):
    Figure, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    Figure.set_size_inches(12, 4, True)
    
    ax1.title.set_text('original')
    ax1.axis('off')
    ax1.imshow(scan[0], cmap=plt.cm.gray)
    
    ax2.title.set_text('corrupted')
    ax2.axis('off')
    ax2.imshow(scan_corrupt[0], cmap=plt.cm.gray)
    
    ax3.title.set_text('predicted')
    ax3.axis('off')
    ax3.imshow(scan_predict[0], cmap=plt.cm.gray)
    
    
    def AnimationFunction(frame): 
        ax1.imshow(scan[frame], cmap=plt.cm.gray)
        ax2.imshow(scan_corrupt[frame], cmap=plt.cm.gray)
        ax3.imshow(scan_predict[frame], cmap=plt.cm.gray)
        
        
    anim = FuncAnimation(Figure, AnimationFunction, frames=scan.shape[0], interval=100)
    
    html = display.HTML(anim.to_jshtml())
    display.display(html)
    
    plt.close()
    
def Load_Training_Images(images_folder:str):
    exclude=[images_folder+'/image.png',images_folder+'image_corrupt.png']
    reconstructed = glob(images_folder + '/*.png')
    reconstructed = [r for r in reconstructed if r not in exclude]
    reconstructed.sort(reverse=True)

    images = []
    for image_path in reconstructed:
        im = plt.imread(image_path)
        images.append(plt.imread(image_path))
    images = np.array(images)
    return images

def Animate_Model_Training(images_folder:str, video_path:str=''):
    
    images, titles = Load_Training_Images(images_folder)
    figure = plt.figure()

    lines_plotted = plt.imshow(images[0], cmap=plt.cm.gray)    

    def show_animation(frame: np.array):
        plt.imshow(images[frame], cmap=plt.cm.gray)
        plt.axis('off')
        if frame==0:
            plt.title('corrupt image')
        elif frame==1:
            plt.title('epoch: 0')
        else:
            title = titles[frame].replace(images_folder + '/', '')
            title = title.replace('.png', '')
            title = title.lstrip('0')
            plt.title('epoch: ' + str(title))

    anim = FuncAnimation(figure, show_animation, frames=images.shape[0], interval=500)

    writervideo = animation.FFMpegWriter(fps=2)
    if video_path != '':
        anim.save(video_path, writervideo)

    video = anim.to_jshtml()
    html = display.HTML(video)
    display.display(html)

    plt.close()