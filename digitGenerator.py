## IMPORTS
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import glob
import imageio

## PARAMETERS
num_epochs = 100
batch_size = 128
image_size = (28, 28)
z_size = 20
mode_z = 'uniform'
tf.random.set_seed(1)
np.random.seed(1)

## MAKE GENERATOR
def make_generator(
    z_size=20,
    output_size=(28, 28, 1),
    n_filters=128,
    n_blocks=2):

    size_factor = 2**n_blocks
    hidden_size = (output_size[0]//size_factor, output_size[1]//size_factor)

    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(z_size,)),
    tf.keras.layers.Dense(units=n_filters*np.prod(hidden_size), use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Reshape((hidden_size[0], hidden_size[1], n_filters)),
    tf.keras.layers.Conv2DTranspose(filters=n_filters, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU()
    ])

    nf = n_filters
    for i in range(n_blocks):
        nf = nf // 2
        model.add(tf.keras.layers.Conv2DTranspose(filters=nf, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(filters=output_size[2], kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model

## MAKE DISCRIMINATOR
def make_discriminator(
     input_size=(28, 28, 1),
     n_filters=64,
     n_blocks=2):

    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_size),
    tf.keras.layers.Conv2D(filters=n_filters, kernel_size=5,strides=(1, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU()
    ])

    nf = n_filters

    for i in range(n_blocks):
        nf = nf*2
        model.add(tf.keras.layers.Conv2D(filters=nf, kernel_size=(5, 5), strides=(2, 2),padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.3))


    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(7, 7), padding='valid'))

    model.add(tf.keras.layers.Reshape((1,)))

    return model

## PREPROCESSING
def preprocess(ex, mode='uniform'):
    ## ACCESS IMAGE AND CONVERT TO TENSOR
    image = ex['image']
    image = tf.image.convert_image_dtype(image, tf.float32)
    ## CONVERT FROM [0,1] TO [-1,1]
    image = image*2 - 1.0

    ## SELECT UNIFORM OR NORMAL DISTRIBUTION FOR Z INPUT
    if mode == 'uniform':
        input_z = tf.random.uniform(shape=(z_size,), minval=-1.0, maxval=1.0)
    elif mode == 'normal':
        input_z = tf.random.normal(shape=(z_size,))
    return input_z, image

## BCE LOSS FUNCTION
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

## DISCRIMINATOR LOSSES
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

## GENERATOR LOSS
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

## OPTIMIZERS
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

## DOWNLOAD AND PREPARE DATASET
mnist_bldr = tfds.builder('mnist')
mnist_bldr.download_and_prepare()
mnist = mnist_bldr.as_dataset(shuffle_files=False)
mnist_trainset = mnist['train']
mnist_trainset = mnist_trainset.map(preprocess)
mnist_trainset = mnist_trainset.shuffle(10000)
mnist_trainset = mnist_trainset.batch(batch_size, drop_remainder=True)

## CREATE MODEL
generator = make_generator()
generator.build(input_shape=(None, z_size))
discriminator = make_discriminator()
discriminator.build(input_shape=(None, np.prod(image_size)))

## CHECKPOINTING
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

## DEFINE TRAINING STEP
def train_step(data):

    input_z, images = data

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(input_z, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

## DEFINE TRAINING LOOP
def train(dataset, epochs):
    for epoch in range(epochs):

        for image_batch in tqdm(dataset,desc="EPOCH {}".format(epoch)):
            train_step(image_batch)

        predictions = generator(test_input, training=False)

        fig = plt.figure(figsize=(8,8))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

## TRAIN
train(mnist_trainset, num_epochs)

## CREATE GIF
anim_file = '/content/drive/My Drive/dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
