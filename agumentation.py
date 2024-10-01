from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_data(images, masks):

    data_gen_args = dict(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    return image_datagen, mask_datagen
