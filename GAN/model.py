from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam


##### Define global parameters

class GAN_MODEL():

    def __init__(self, latent_dim, img_shape_d, img_shape_l):

        self.latent_dim = latent_dim
        self.img_shape_d = img_shape_d
        self.img_shape_l = img_shape_l
        

    ##### Function for defining generator

    def build_generator(self, n_filters = 64, growth_factor = 2):
        
        ##### Noise input
        noise = Input(shape=(self.latent_dim,))
        
        ##### black input
        image_L = Input(shape=self.img_shape_l)

        ##### Create combined input of noise and L
        noise_h = Dense(256 * 8 * 8, activation="relu")(noise)
        noise_h = Reshape((128, 128, 1))(noise_h)
        combined_input = concatenate([noise_h, image_L])
        
        ##### Define Model
        
        # Encoding
        conv1 = Conv2D(n_filters, (4,4), strides=(1,1), kernel_initializer="he_normal", padding="same")(combined_input)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        conv1 = BatchNormalization()(conv1)

        n_filters *= growth_factor
        conv2 = Conv2D(n_filters, (4,4), strides=(2,2), kernel_initializer="he_normal", padding="same")(conv1)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        conv2 = BatchNormalization()(conv2)

        n_filters *= growth_factor
        conv3 = Conv2D(n_filters, (4,4), strides=(2,2), kernel_initializer="he_normal", padding="same")(conv2)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        conv3 = BatchNormalization()(conv3)
            
        n_filters *= growth_factor
        conv4 = Conv2D(n_filters, (2,2), strides=(2,2), kernel_initializer="he_normal", padding="same")(conv3)
        conv4 = LeakyReLU(alpha=0.2)(conv4)
        conv4 = BatchNormalization()(conv4)

        n_filters *= growth_factor
        conv5 = Conv2D(n_filters, (1,1), strides=(2,2), kernel_initializer="he_normal", padding="same")(conv4)
        conv5 = LeakyReLU(alpha=0.2)(conv5)
        conv5 = BatchNormalization()(conv5)
        
        # Latent space
        mid_layer = Conv2D(n_filters, (1,1), strides=(2,2), kernel_initializer="he_normal", padding="same")(conv5)
        mid_layer = LeakyReLU(alpha=0.2)(mid_layer)

        # Decoding
        n_filters //= growth_factor
        deconv1 = Conv2DTranspose(n_filters, (1,1), strides=(2,2), kernel_initializer="he_normal", padding="same")(mid_layer)
        deconv1 = LeakyReLU(alpha=0.2)(deconv1)
        deconv1 = BatchNormalization()(deconv1)
        concat1 = concatenate([deconv1, conv5])
        
        n_filters //= growth_factor
        deconv2 = Conv2DTranspose(n_filters, (2,2), strides=(2,2), kernel_initializer="he_normal", padding="same")(concat1)
        deconv2 = LeakyReLU(alpha=0.2)(deconv2)
        deconv2 = BatchNormalization()(deconv2)
        concat2 = concatenate([deconv2, conv4])

        n_filters //= growth_factor
        deconv3 = Conv2DTranspose(n_filters, (4,4), strides=(2,2), kernel_initializer="he_normal", padding="same")(concat2)
        deconv3 = LeakyReLU(alpha=0.2)(deconv3)
        deconv3 = BatchNormalization()(deconv3)
        concat3 = concatenate([deconv3, conv3])
        
        n_filters //= growth_factor
        deconv4 = Conv2DTranspose(n_filters, (4,4), strides=(2,2), kernel_initializer="he_normal", padding="same")(concat3)
        deconv4 = LeakyReLU(alpha=0.2)(deconv4)
        deconv4 = BatchNormalization()(deconv4)
        concat4 = concatenate([deconv4, conv2])
        
        n_filters //= growth_factor
        deconv5 = Conv2DTranspose(n_filters, (4,4), strides=(2,2), kernel_initializer="he_normal", padding="same")(concat4)
        deconv5 = LeakyReLU(alpha=0.2)(deconv5)
        deconv5 = BatchNormalization()(deconv5)
        concat5 = concatenate([deconv5, conv1])
        
        
        ##### Finalizing the layer
        deconv6 = Conv2DTranspose(n_filters, (4,4), strides=(1,1), kernel_initializer="he_normal", padding="same")(concat5)
        deconv6 = LeakyReLU(alpha=0.2)(deconv6)
        deconv6 = BatchNormalization()(deconv6)

        ab_layer = Conv2D(2, kernel_size=(4,4), strides=(1,1), kernel_initializer="he_normal", padding="same", activation="tanh")(deconv6)

        img = Concatenate(axis=3)([image_L, ab_layer])
        
        ##### define the model
        model_final = Model([noise, image_L], img)
        model_final.summary()
        
        return model_final


    ##### Function for defining discriminator

    def build_discriminator(self, n_filters = 16, growth_factor = 2):

        ##### Created input
        img = Input(shape=self.img_shape_d)
        
        ##### Define model
        model = Conv2D(n_filters, kernel_size=(4,4), strides=2, kernel_initializer='he_normal', padding="same")(img)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dropout(0.30)(model)
        
        n_filters *= growth_factor
        model = Conv2D(n_filters, kernel_size=(4,4), strides=2, kernel_initializer='he_normal', padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dropout(0.30)(model)

        n_filters *= growth_factor
        model = Conv2D(n_filters, kernel_size=(4,4), strides=2, kernel_initializer='he_normal', padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dropout(0.30)(model)
        
        n_filters *= growth_factor
        model = Conv2D(n_filters, kernel_size=(4,4), strides=1, kernel_initializer='he_normal', padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dropout(0.30)(model)
        
        n_filters *= growth_factor
        model = Conv2D(n_filters, kernel_size=(4,4), strides=1, kernel_initializer='he_normal', padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dropout(0.30)(model)
        
        n_filters *= growth_factor
        model = Conv2D(n_filters, kernel_size=(4,4), strides=1, kernel_initializer='he_normal', padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Dropout(0.30)(model)
        
        model = Flatten()(model)
        model = Dense(1, activation="sigmoid")(model)

        model_final = Model(img, model)
        model_final.summary()

        return model_final