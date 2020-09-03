# Digit Generator

The aim of this project was to use a DCGAN to produce fake handwritten digits.

### Quick DCGAN Overview

The objective of the network is to produce a fake image from some random vector (z).

z &#8594; Generator :right_arrow: Fake image

In order to imporve the generator model we must create a discriminator network to distinguish between real and generated fake images.

z :right_arrow: Generator :right_arrow: Fake image :right_arrow: Discriminator :right_arrow: Generator loss

However we must also train our discriminator network by passing real and fake images to the discriminator.

Real and fake images :right_arrow: Discriminator :right_arrow: Discriminator loss

One training step can be sumarised as follows
1. Calculate the generator loss
2. Update the generator weights
3. Calculate the discriminator loss
4. Update the discriminator weights

Hence the combined network will learn to produce more and more realistic images.

### Results

Here is the network learning to create handwritten digits!

![Results GIF](learningToWrite.gif)
