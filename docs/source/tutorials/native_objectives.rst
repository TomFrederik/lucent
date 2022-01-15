.. _native_objectives:

=================
Native Objectives
=================

This page covers all of the pre-existing objectives that come with installing Lucent.

.. contents:: Table of Contents



``channel`` objective
=====================


The channel objective is the default objective in Lucent. It will take a specific layer and a specific channel and will optimize the input image such that the
mean activation over the whole feature map in this channel is optimized. That is, if the latent feature map is of size ``H x W`` for each channel, then this 
objective will average the activation over the ``H * W`` positions in the latent feature map for the specified channel.

``neuron`` objective
====================

The neuron objective is like the channel objective but instead of averaging over the ``H * W`` latent feature map positions, it will take a specific ``x-y`` position
in the latent feature map and only optimize for the activation at that position. By default, it will take the center position.


.. admonition:: Question

   What would you expect the difference in the resulting images to be, when comparing the ``channel`` and ``neuron`` objective?
   
   .. raw:: html

      <details>
      <summary><a>Answer</a></summary>
 
   
   Since we know that convnets are translation equivariant we would expect the same feature to excite the same channel at different positions in the image.
   Averaging over the latent feature map would then result in the same feature being repeated over and over again in the input image, whereas for the neuron
   objective we would expect only a single instance of the feature.
   
   .. raw:: html

      </details>
   
   TODO: add images
   
   
``weight`` objectives
=====================

Instead of specifying a single neuron or channel we can also give a linear combination of neurons/channels, by providing a weight vector of length ``num_channels``.
The respective objectives are ``channel_weight`` and ``neuron_weight``.











