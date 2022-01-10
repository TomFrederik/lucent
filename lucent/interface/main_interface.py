import time
import os

# from lucent.optvis.objectives import channel
from lucent.optvis.render import render_vis, tensor_to_img_array
import numpy as np
from PIL import Image
import streamlit as st
import torch

from lucent.interface.utils import join_layer_channel, update_image_db, display_image, init


def update_identifier():
    st.session_state.identifier = join_layer_channel(st.session_state.layer, st.session_state.channel)


if 'layer_names' not in st.session_state:
        st.session_state['layer_names'] = []

with st.sidebar:
    with st.form('config'):
        st.write('## Config')
        
        st.selectbox(
            'Model', 
            options=[
                None,
                'resnet18',
                'alexnet',
                'squeezenet1_0',
                'vgg16',
                'densenet161',
                'inception_v3',
                'googlenet',
                'shufflenet_v2_x1_0',
                'mobilenet_v2',
                'mobilenet_v3_small',
                'mobilenet_v3_large',
                'resnext50_32x4d',
                'wide_resnet50_2',
                'mnasnet1_0',
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
            index=0,
        )
            

        st.text_input('Data directory', value='/home/aric/Pictures/', key='datadir')
        st.checkbox('Load images from data dir', value=True, key='load_data')
        
        
        submitted = st.form_submit_button("Save config")
        if submitted:
            print(f'\n{st.session_state.layer = }, {st.session_state.channel = }\n')
            st.session_state.model = init(st.session_state.model_name)
            st.write('Config saved!')

    # this should have a disabled keyword but somehow doesn't yet --> maybe code on github needs to be pushed to package manager first
    st.checkbox("Save images to data dir (won't work if loading images)", value=False, key='save_data') 

    st.selectbox('layer', options=st.session_state.layer_names, key='layer', on_change=update_identifier, index=0)
    st.text_input('channel', value='1', key='channel', on_change=update_identifier)


# init and update data base of features
if 'database' not in st.session_state:
    st.session_state['database'] = dict()


# init identifier
if 'identifier' not in st.session_state:
    st.session_state['identifier'] = None

if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'layer' not in st.session_state:
    st.session_state['layer'] = None

if st.session_state.load_data:
    update_image_db(st.session_state.datadir, st.session_state.model_name)

st.button('Generate/Load image', on_click=display_image, args=(st.session_state.model, st.session_state.identifier,))