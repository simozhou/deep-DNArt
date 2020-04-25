import numpy as np
import time
import tensorflow as tf
from tf.Data import Dataset, Iterator

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

class ElapsedTimer(object):
    """Displays state of execution on the CLI"""
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    """Class object for the GAN network"""
    def __init__(self, img_rows=256, img_cols=256, channel=3):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 32
        dropout = 0.4
        # In: ?, depth = 1
        # Out: ?, depth=32
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        # Out: depth=64
        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        # Out: depth=88
        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        # Out: depth=128
        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        # TODO len(vector of SNPs)
        depth = 64
        dim = 32
        # In: ?
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=40000))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 256 x 256 x 3 RGB image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        """builds the discriminator model with dedicated optmizer and loss function"""
        if self.DM:
            return self.DM
        # optimizable parameters (learning rate and decay)
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        # optimizable parameters (learning rate and decay)
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class Artsy_DCGAN(object):
    def __init__(self, img_dir):
        self.img_rows = 256
        self.img_cols = 256
        self.channel = 3
        # we set the batch size to half of the actual one,
        # so that the final batch (real images + noise generated becomes 128)
        self.batch_size = 64
        # DATA INPUT: we generate 256x256 images
        train_datagen = ImageDataGenerator(
        rescale=1/255,
        zoom_range=0.2,
        fill_mode='nearest'
		)

        train_generator = train_datagen.flow_from_directory(
        img_dir,
        classes=['landscape'],
        target_size=(256, 256),
        batch_size=self.batch_size,
        class_mode=None,
        color_mode="rgb"
        )

        self.x_train = train_generator
        # self.x_train = self.x_train.reshape(-1, self.img_rows,\
        #  	self.img_cols, 1).astype(np.float32)

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, save_interval=0):
        noise_input = None
        if save_interval>0:
            # noise should have the same size of our data!
            noise_input = np.random.binomial(1, 0.05, size=[16, 40000])
        # for each batch
        for images_batch in self.x_train:
            # this might need ad additional channel
            images_train = images_batch
            noise = np.random.binomial(1, 0.05, size=[self.batch_size, 40000])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*self.batch_size, 1])
            y[self.batch_size:, :] = 0
            # we train the discriminator first (teaching the police how to do its job)
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([self.batch_size, 1])
            noise = np.random.binomial(1, 0.05, size=[self.batch_size, 40000])
            # then we let the police teach the generator how to create good images
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'artsy.png'
        if fake:
            if noise is None:
                noise = np.random.binomial(1, 0.05, size=[samples, 40000])
            else:
                filename = "artsy_%d.png" % step
            images = self.generator.predict(noise)
        else:
        	# TODO fix this, train_generator has no shape attribute
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    # test run
    artsy_dcgan = Artsy_DCGAN(img_dir="/home/ubuntu/simone/test/")
    timer = ElapsedTimer()
    artsy_dcgan.train(train_steps=10000, save_interval=500)
    timer.elapsed_time()
    artsy_dcgan.plot_images(fake=True)
    # artsy_dcgan.plot_images(fake=False, save2file=True)