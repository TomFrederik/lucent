import time
import os

import torchvision.models as models
# from lucent.optvis.objectives import channel
from lucent.optvis.render import render_vis, tensor_to_img_array
from lucent.modelzoo.util import get_model_layers
import numpy as np
from PIL import Image
import streamlit as st
import torch


@st.experimental_singleton
def init(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_name != 'Other (specify below)':
        model = getattr(models, model_name)
        print(model)
        model = model(pretrained=True)
    model.to(device).eval()

    st.session_state.layer_names = get_model_layers(model)

    return model

# @st.experimental_memo # if you don't have this it regenerates the image every time you change something --> longterm replace this with a DB lookup?
def generate_image(identifier):
    if st.session_state.load_data:
        image = check_database(identifier)
    else:
        image = None
    if image is None:
        image = (render_vis(model, identifier, show_image=False, thresholds=(64,))[0][0] * 255).astype(np.uint8)
        image = Image.fromarray(image)
        layer, name = split_identifier(identifier)
        if st.session_state.save_data:
            layer_path = os.path.join(st.session_state.datadir, layer)
            os.makedirs(layer_path, exist_ok=True)
            image.save(os.path.join(layer_path, name) + '.jpg')
        
        # add newly generated image to database
        st.session_state.database = st.session_state.database | {layer: {name: image}}

    return image

def display_image(identifier):
    st.session_state['image'] = generate_image(identifier)
    st.image(st.session_state.image)

@st.experimental_memo
def load_images(datadir):
    add_to_db = dict()
    layers = next(os.walk(datadir))[1]
    for layer in layers:
        layer_image_paths = next(os.walk(os.path.join(datadir, layer)))[-1]
        layer_name = layer.split('/')[-1]
        for image_path in layer_image_paths:
            image = Image.open(os.path.join(datadir, layer, image_path))
            image_id = image_path.split('.')[0]
            if layer_name not in add_to_db:
                add_to_db[layer_name] = dict()
            add_to_db[layer_name][image_id] = image
    return add_to_db

def update_image_db(datadir):
    add_to_db = load_images(datadir)
    st.session_state.database = st.session_state.database | add_to_db # need python 3.9+ for this

def check_database(identifier):
    for layer, subdict in st.session_state.database.items():
        for name, image in subdict.items():
            if join_layer_channel(layer, name) == identifier:
                return image
    return None

def join_layer_channel(layer, name):
    return ':'.join([layer, name])

def split_identifier(identifier):
    return identifier.split(':')

if 'layer_names' not in st.session_state:
        st.session_state['layer_names'] = []

with st.sidebar:
    with st.form('config'):

        st.selectbox(
            'Model', 
            options=[
                'resnet18',
                'alexnet',
                'squeezenet',
                'vgg16',
                'densenet',
                'inceptionV3',
                'googlenet',
                'shufflenet',
                'mobilenet_v2',
                'mobilenet_v3_small',
                'mobilenet_v3_large',
                'resnext50_32x4d',
                'wide_resnet50_2',
                'mnasnet',
                'efficientnet_b0',
                'efficientnet_b1',
                'efficientnet_b2',
                'efficientnet_b3',
                'efficientnet_b4',
                'efficientnet_b5',
                'efficientnet_b6',
                'efficientnet_b6',
                'efficientnet_b7',
                'regnet_y_400mf',
                'regnet_y_800mf',
                'regnet_y_1_6gf',
                'regnet_y_3_2gf',
                'regnet_y_8gf',
                'regnet_y_16gf',
                'regnet_y_32gf',
                'regnet_x_400mf',
                'regnet_x_800mf',
                'regnet_x_1_6gf',
                'regnet_x_3_2gf',
                'regnet_x_8gf',
                'regnet_x_16gf',
                'regnet_x_32gf',
                'Other (specify below)',
            ],
            key='model_name',
        )
            

        st.write('## Config')
        st.selectbox('layer', options=st.session_state.layer_names, key='layer')
        st.text_input('channel', value='476', key='channel')
        st.text_input('Data directory', value='/home/aric/Pictures/', key='datadir')
        st.checkbox('Load images from data dir', value=True, key='load_data')
        
        
        submitted = st.form_submit_button("Save config")
        if submitted:
            print(f'\n{st.session_state.layer = }, {st.session_state.channel = }\n')
            st.session_state.identifier = join_layer_channel(st.session_state.layer, st.session_state.channel)
            st.write('Config saved!')

    # this should have a disabled keyword but somehow doesn't yet --> maybe code on github needs to be pushed to package manager first
    st.checkbox("Save images to data dir (won't work if loading images)", value=False, key='save_data') 

model = init(st.session_state.model_name)

# init and update data base of features
if 'database' not in st.session_state:
    st.session_state['database'] = dict()

if st.session_state.load_data:
    update_image_db(st.session_state.datadir)

st.button('Generate/Load image', on_click=display_image, args=(st.session_state.identifier,))