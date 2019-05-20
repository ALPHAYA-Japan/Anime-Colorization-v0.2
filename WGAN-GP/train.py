##### Import necessary module

from sklearn.model_selection import train_test_split
from keras.layers.merge import _Merge
from functools import reduce, partial
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam


##### Import custom functions

from data_preprocessing import *
from model import WGAN_GP_MODEL


##### Functon for Wasserstein loss and gradient penalty

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""
    return -K.mean(y_true * y_pred)

# define batch_size and img_shape_d here... need to improve the code
batch_size = 32
img_shape_d = (128, 128, 3)

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((batch_size,) + img_shape_d)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1]) # alpha here represents an epsilon in the paper


def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


##### Function for preparing dataset and training the model

def train(size_col, size_row, latent_dim, dir_name, epochs, save_interval, batch_size, n_critic, print_process = True, visualization = True):

    ##### Clear session for initialization
    K.clear_session()

    input_shape = (size_row, size_col)
    img_shape_d = (input_shape[0], input_shape[1], 3)
    img_shape_l = (input_shape[0], input_shape[1], 1)


    ##### Preparing dataset
    print("==============================")
    print("===== Preparing dataset for training model ...")
    print("==============================")

    image_file = os.listdir(dir_name)

    train_image = []
    train_label = []

    test_image = []
    test_label = []

    image_size = (size_row, size_col)


    # split into train and test here
    train_index, test_index = train_test_split(np.arange(len(image_file)), test_size = 0.1)

    train_list = np.array(image_file)[list(train_index)]
    test_list = np.array(image_file)[list(test_index)]

    # train case
    for train_num in range(len(train_list)):

        file_chosen = dir_name + '/{}'.format(train_list[train_num])

        if plt.imread(file_chosen).shape[2] == 3:
            image_chosen = cv2.resize(plt.imread(file_chosen), image_size)

            train_image_indiv, train_label_indiv = create_feature_label_image(image_chosen)

            train_image.append(train_image_indiv)
            train_label.append(train_label_indiv)

    # test case
    for test_num in range(len(test_list)):

        file_chosen = dir_name + '/{}'.format(test_list[test_num])

        if plt.imread(file_chosen).shape[2] == 3:
            image_chosen = cv2.resize(plt.imread(file_chosen), image_size)

            test_image_indiv, test_label_indiv = create_feature_label_image(image_chosen)

            test_image.append(test_image_indiv)
            test_label.append(test_label_indiv)


    # Converting to numpy
    train_image = np.array(train_image)
    test_image = np.array(test_image)

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    train_feature = train_image.reshape(train_image.shape[0], size_row, size_col,1) 
    test_feature = test_image.reshape(test_image.shape[0], size_row, size_col,1)

    x_feature = np.concatenate([train_feature, test_feature])
    x_label = np.concatenate([train_label, test_label])

    print("==============================")
    print("===== Dataset has been prepared!")
    print("==============================")

    ##### Define model
    print("==============================")
    print("===== Training the model ...")
    print("==============================")

    model_class = WGAN_GP_MODEL(latent_dim, img_shape_d, img_shape_l)

    # Build the generator and discriminator
    K.clear_session()

    print("===== Generator =====")
    generator = model_class.build_generator()

    print("===== Discriminator =====")
    discriminator = model_class.build_discriminator()

    # Defining input
    real_img = Input(shape = img_shape_d)
    img_l = Input(shape = (img_shape_l))
    z_disc = Input(shape = (latent_dim,))
    optimizer_d = Adam(0.0001, 0.5)

    # Using input, get fake and valid image
    fake_img_d = generator([z_disc, img_l])
    valid = discriminator(real_img) # discriminator output layer of real images
    fake = discriminator(fake_img_d) # discriminator output layer of fake images

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage()(inputs = [real_img, fake_img_d]) # calculation interpolation xhat 
    # Determine validity of weighted sample
    validity_interpolated = discriminator(interpolated_img) # D(xhat)

    # Use Python partial to provide loss function with additional 'averaged_samples' argument
    partial_gp_loss = partial(gradient_penalty_loss,
                    averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

    optimizer_d = Adam(0.0001, 0.5)

    #-------------------------------
    # Construct Computational Graph
    #         for Discriminator
    #-------------------------------

    generator.trainable = False # after setting generator false, create discriminator model so that only generator can be trained

    discriminator_model = Model(inputs=[real_img, z_disc, img_l],
                        outputs=[valid, fake, validity_interpolated])

    discriminator_model.compile(loss=[wasserstein_loss,
                                    wasserstein_loss,
                                    partial_gp_loss],
                                optimizer=optimizer_d,
                                loss_weights=[1, 1, 10])

    #-------------------------------
    # Construct Computational Graph
    #         for Generator
    #-------------------------------

    # For the generator we freeze the discriminator's layers
    discriminator.trainable = False # after setting discriminator false, create generator model so that only generator can be trained
    generator.trainable = True

    # Sampled noise for input to generator
    z_gen = Input(shape=(latent_dim,))

    # Generate images based of noise
    fake_img_g = generator([z_gen, img_l])

    # Discriminator determines validity
    valid = discriminator(fake_img_g)

    # Defines generator model
    optimizer_g = Adam(0.0001, 0.5)
    generator_model = Model([z_gen, img_l], valid)
    generator_model.compile(loss=wasserstein_loss, optimizer=optimizer_g)


    ##### Labeling
    valid = np.ones((batch_size, 1))
    fake =  -np.ones((batch_size, 1))
    dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

    ##### Get sample images
    sample_index = np.random.choice(x_feature.shape[0], batch_size)
    sample_channel = x_feature[sample_index]
    sample_images(sample_channel)


    ##### Training
    g_losses = []
    d_losses = []

    for epoch in range(epochs):

        for _ in range(n_critic):
        
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch-size number of images
            idx = np.random.randint(0, x_feature.shape[0], batch_size)
            l_channel = x_feature[idx]
            ab_channel = x_label[idx] 
            img_lab = np.concatenate([l_channel, ab_channel], axis = 3)
            
            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, latent_dim))        
            
            # Train the discriminator
            d_loss = discriminator_model.train_on_batch([img_lab, noise, l_channel],
                                                        [valid, fake, dummy])


        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = generator_model.train_on_batch([noise, l_channel], valid)

        g_losses.append(g_loss)
        d_losses.append(d_loss[0])
        

        if print_process:
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            noise = np.random.normal(0, 1, (25, latent_dim))
            sample_predicted = generator.predict([noise, l_channel]) * 128
            sample_images(sample_predicted)
        
    print("==============================")
    print("===== Training has been completed!")
    print("==============================")


    ##### Visualizing the result
    
    if visualization:

        print("==============================")
        print("===== Visualization!")
        print("==============================")

        # Images
        sample_predicted = generator.predict([noise, sample_channel])*128
        sample_images(sample_predicted)

        # Plots
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses,label="G")
        plt.plot(d_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    ##### Return input data, sample index (for visualization, and generator

    return x_feature, generator