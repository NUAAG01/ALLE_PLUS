import os
import numpy as np
import cv2
import imgaug.augmenters as iaa
 
 
seq = iaa.Sequential([
        # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
        #iaa.Sometimes(
        #    0.5,
        #    iaa.GaussianBlur(sigma=(0, 0.5))
        #),
        #iaa.WithBrightnessChannels(
        #    iaa.Add((-50, 50)), to_colorspace=[iaa.CSPACE_Lab, iaa.CSPACE_HSV]),
        iaa.Multiply((0.1, 1.8))
        #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
        #iaa.Multiply((0.8, 1.2), per_channel=0.2),
], random_order=True) # apply augmenters in random order

class MiniBatchLoader(object):

 
    def __init__(self, train_path, test_path, image_dir_path, crop_size):
 
        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_paths(test_path, image_dir_path)
        #print("this is the training")
        self.crop_size = crop_size
 
    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            #print(line)
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path
 
    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c
 
    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        #print(txt_path)
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            #print("Reading")
            cs.append(pair)
        return cs
 
    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True)
 
    def load_testing_data(self, indices):
        return self.load_data(self.testing_path_infos, indices)
 
    # test ok
    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 3

        if augment:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            #ys = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            for i, index in enumerate(indices):
                path = path_infos[index]
                
                img = cv2.imread(path)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                h, w, _ = img.shape

                if np.random.rand() > 0.5:
                    img = np.fliplr(img)

                if np.random.rand() > 0.5:
                    angle = 10*np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
                    img = cv2.warpAffine(img,M,(w,h))

                rand_range_h = h-self.crop_size
                rand_range_w = w-self.crop_size
                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)
                img = np.transpose(img[y_offset:y_offset+self.crop_size, x_offset:x_offset+self.crop_size],(0,1,2))
                #img_aug = seq(image = img)
                #img_aug = np.transpose(img_aug,(2,0,1))
                img = np.transpose(img, (2,0,1))
                xs[i, :, :, :] = (img/255).astype(np.float32)
                #ys[i, :, :, :] = (img_aug/255).astype(np.float32)

        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                path = path_infos[index]
                
                img = cv2.imread(path)
                h, w, _ = img.shape
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                '''
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)

                if np.random.rand() > 0.5:
                    angle = 10*np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
                    img = cv2.warpAffine(img,M,(w,h))
                '''
                
                
            
            rand_range_h = h-self.crop_size
            rand_range_w = w-self.crop_size
           # x_offset = np.random.randint(rand_range_w)
            #y_offset = np.random.randint(rand_range_h)
            #img = np.transpose(img[y_offset:y_offset+self.crop_size, x_offset:x_offset+self.crop_size],(0,1,2))
            #img_aug = seq(image = img)
            xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            #ys = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            xs[0, :, :, :] = np.transpose((img/255).astype(np.float32),(2,0,1))
            #ys[0, :, :, :] = np.transpose((img_aug/255).astype(np.float32),(2,0,1))

        else:
            raise RuntimeError("mini batch size must be 1 when testing")
 
        return xs

    def data_denoise(self,img):
        h, w, c = img.shape
        (img_b, img_g, img_r) = cv2.split(img)
        xs_r = np.zeros((1, 1, h, w)).astype(np.float32)
        xs_g = np.zeros((1, 1, h, w)).astype(np.float32)
        xs_b = np.zeros((1, 1, h, w)).astype(np.float32)
        xs_b[0, 0, :, :] = (img_b / 255).astype(np.float32)
        xs_g[0, 0, :, :] = (img_g / 255).astype(np.float32)
        xs_r[0, 0, :, :] = (img_r / 255).astype(np.float32)
        return xs_b, xs_g, xs_r