import time
import os

# from lucent.optvis.objectives import channel
from lucent.optvis.render import render_vis, tensor_to_img_array
import numpy as np
from PIL import Image
import streamlit as st
import torch

from lucent.interface.utils import update_image_db, display_image, init, display_database, generate_layer_features, create_model_list


def layer_change():
    st.session_state.identifier = st.session_state.layer
    update_display()

def update_display():
    print('Updating display..')
    if st.session_state.display == 'Layer':
        display_database(st.session_state.layer)
    elif st.session_state.display == 'Database':
        display_database()

def changed_nesting_depth():
    pass #TODO


st.set_page_config(layout="wide")


if 'layer_names' not in st.session_state:
        st.session_state['layer_names'] = []

if 'dependence_graph' not in st.session_state:
        st.session_state['dependence_graph'] = []

if 'dep_graph_depth' not in st.session_state:
    st.session_state['dep_graph_depth'] = 1

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
            st.session_state.model = init(st.session_state.model_name)
            st.write('Config saved!')

    # this should have a disabled keyword but somehow doesn't yet --> maybe code on github needs to be pushed to package manager first
    st.checkbox("Save images to data dir", value=False, key='save_data') 

    st.selectbox('layer', options=st.session_state.layer_names, key='layer', on_change=layer_change, index=0)
    #TODO apparently slider has to have min != max value -> only enable slider when more than one depth?
    st.slider('Nesting Depth', min_value=1, max_value=st.session_state.dep_graph_depth+1, value=1, step=1, key='nesting_depth', on_change=changed_nesting_depth)
    st.button('Generate Layer Features', on_click=generate_layer_features, args=(st.session_state.model, st.session_state.layer))
    st.select_slider('Display ', options=['Layer', 'Database'], value='Layer', key='display', on_change=update_display)



if st.session_state.load_data:
    print('\nloading data from database...\n')
    update_image_db(st.session_state.datadir, st.session_state.model_name)


# display_database(st.session_state.layer)