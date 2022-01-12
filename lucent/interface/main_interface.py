import time
import os

# from lucent.optvis.objectives import channel
from lucent.optvis.render import render_vis, tensor_to_img_array
import numpy as np
from PIL import Image
import streamlit as st
import torch

from lucent.interface.utils import join_layer_channel, update_image_db, display_image, init, create_model_list


def update_identifier():
    st.session_state.identifier = join_layer_channel(st.session_state.layer, st.session_state.channel)


if 'layer_names' not in st.session_state:
        st.session_state['layer_names'] = []

with st.sidebar:
    with st.form('config'):
        st.write('## Config')
        
        st.selectbox(
            'Model', 
            options=create_model_list(),
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