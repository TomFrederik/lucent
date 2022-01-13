.. _first_steps:

=====================
First steps in Lucent
=====================

.. contents:: Table of Contents


.. note:: 
    This page assumes you already have :ref:`installed <installation>`_ Lucent

.. note::
    This page is still WIP. While it is under construction please check out the existing colab notebooks on the original project's github `main page <https://github.com/greentfrapp/lucent>`_.


Graphical interface vs. script 
==============================

Lucent can be interacted with either indirectly via the graphical interface or directly via a python script by calling the appropriate lucent.optvis methods.

When you are just starting out with interpretability we recommend having a look at the graphical interface first and then transitioning to optvis later if you want to change some of the default settings, add your own custom objective function or what have you.


Graphical Interface
===================

The graphical interface is still WIP, but you can check it out by running

.. parsed-literal::
    cd $LUCENT_PATH/lucent/interface
    streamlit run investigate_layer.py


This should then open a tab in your browser looking like this

.. image:: ./images/investigate_layer_startup.png
  :width: 1280
  :alt: investigate_layer_startup

You can now select a model of your choice, either from the torchvision modelzoo or upload your own model. 
If you wish to load from an old session, you can specify the data directory and tick the 'Load images from data dir' checkbox.

Click 'Save Config'. Lucent should automatically detect all relevant layers for you and list them in the layer drop menu.
Now you can generate the features for each layer by selecting the layer and clicking 'Generate Layer Features'.

If you select 'Display Database', all of the loaded and generated images for the selected model will be displayed.

Lucent comes with a couple of predefined interfaces geared towards investigating different phenomena. You can check them out under the folder interface.


Lucent via Script
=================

We recommend using an interactive environment for this, such as your own jupyter notebook or a Google Colab.

If you are running the code in a colab, we first need to install lucent:

..
    TODO: make sure this actually works on colab

.. code-block:: python

    !pip install --quiet git+https://github.com/TomFrederik/lucent.git

Now, let's import torch and lucent, and set the device variable. 

.. code-block:: python

    import torch
    from lucent.optvis import render, param, transform, objectives

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

We will now load the InceptionV1 model (also known as GoogLeNet), but you could also use any other image-based network here.
We will send it to the device and set it to eval mode to avoid unnecessary computations and disable any potential dropouts.

Please note that visualization can be painfully slow if you are not using a GPU. Colab provides (limited) access to free GPUs so check them out if you do not have a GPU yourself.

.. code-block:: python

    from torchvision.models import inceptionv1
    model = inceptionv1(pretrained=True)
    _ = model.to(device).eval() # the underscore prevents printing the model if it's the last line in a ipynb cell


Feature visualization
---------------------

Now that we have our model we will start of with the bread and butter of mechanistic interpretability: feature visualization.

The core idea is to optimize the input image to the network such that a certain neuron or channel gets maximally excited. 

How would that help with understanding what network is doing? How could that give us misleading results? Think about it for a minute if it isn't immediately obvious.

.. raw:: html

   <details>
   <summary><a>Answer</a></summary>

Optimizing the input to maximally excite a neuron produces a sort of super-stimulus. It establishes one direction of causality, i.e. ... #TODO
However, this method usually produces images that are very different from the data distribution. We might be worried that it picks up on 
spurious correlations instead of reflecting what the neuron does when it encounters real images.

.. raw:: html

   </details>

In order to perform feature visualization we have to specify an objective function with respect to which we will optimize the input image.

The default of render.render_vis is to assume you gave it a description of the form 'layer:channel' and want it to optimize the whole feature map of the channel.

For example,

.. code-block:: python

    list_of_images = render.render_vis(model, "mixed4a:476") # list of images has one element in this case

Now, what if you don't know the names of all the layers in your network? Lucent has you covered, with its get_model_layers method


.. code-block:: python

    from lucent.modelzoo.util import get_model_layers

    layer_names, dependency_graph = get_model_layers(model)


