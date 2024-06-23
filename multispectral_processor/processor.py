import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def process_multispectral_data(base_dir, channels, train_images, test_images):
    os.makedirs(train_images, exist_ok=True)
    os.makedirs(test_images, exist_ok=True)

    image_names = os.listdir(os.path.join(base_dir, channels[0]))

    images = []
    filenames = []

    for image_name in image_names:
        channel_images = []
        for channel in channels:
            image_path = os.path.join(base_dir, channel, image_name)
            img = Image.open(image_path)
            img_array = np.array(img)
            channel_images.append(img_array)

        multispectral_image = np.stack(channel_images, axis=-1)
        images.append(multispectral_image)
        filenames.append(image_name)

    images = np.array(images)
    X_train, X_test, y_train, y_test = train_test_split(images, filenames, test_size=0.2, random_state=42)

    for img, name in zip(X_train, y_train):
        multispectral_image_pil = Image.fromarray(img.astype('uint8'))
        save_path = os.path.join(train_images, os.path.splitext(name)[0] + '.png')
        multispectral_image_pil.save(save_path, format='PNG')

    for img, name in zip(X_test, y_test):
        multispectral_image_pil = Image.fromarray(img.astype('uint8'))
        save_path = os.path.join(test_images, os.path.splitext(name)[0] + '.png')
        multispectral_image_pil.save(save_path, format='PNG')

    print("Multispectral images have been created and saved as PNG format.")
