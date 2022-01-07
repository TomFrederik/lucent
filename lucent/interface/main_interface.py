import time
import os

from lucent.modelzoo.inceptionv1.InceptionV1 import InceptionV1
# from lucent.optvis.objectives import channel
from lucent.optvis.render import render_vis, tensor_to_img_array
import numpy as np
from PIL import Image
import streamlit as st
import torch


@st.experimental_singleton
def init():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = InceptionV1(pretrained=True)
    model.to(device).eval()
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

        print(st.session_state.database)
    
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
        print(f"{layer_image_paths}")
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

model = init()

# with st.sidebar:

with st.form('config'):
    st.write('## Config')
    st.text_input('layer:channel', value='mixed4a:476', key='channel')
    st.text_input('Data directory', value='/home/aric/Pictures/', key='datadir')
    st.checkbox('Load images from data dir', value=False, key='load_data')
    st.checkbox('Save images to data dir', value=False, key='save_data')
    
    submitted = st.form_submit_button("Save config")
    if submitted:
        st.write('Config saved!')

# init and update data base of features
if 'database' not in st.session_state:
    st.session_state['database'] = dict()

if st.session_state.load_data:
    update_image_db(st.session_state.datadir)

button = st.button('Generate new image', on_click=display_image(st.session_state.channel))