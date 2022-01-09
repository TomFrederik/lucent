import os
from typing import Optional, Dict, List

import numpy as np
from PIL import Image
import streamlit as st
import torch
import torchvision.models as models

from lucent.modelzoo.util import get_model_layers
from lucent.optvis import param
from lucent.optvis.render import render_vis
from lucent.optvis.objectives import Objective, channel

def display_image(
    model: Optional[torch.nn.Module] = None, 
    identifier: Optional[str] = None,
) -> None:
    if identifier is not None and model is not None:
        st.session_state['image'] = generate_image(model, identifier)
        st.image(st.session_state.image)

def generate_image(model, identifier):
    if st.session_state.load_data:
        image = check_database(identifier)
    else:
        image = None
    if image is None:
        image = (render_vis(model, identifier, show_image=False, thresholds=(64,))[0][0] * 255).astype(np.uint8)
        image = Image.fromarray(image)
        layer, name = split_identifier(identifier)
        if st.session_state.save_data:
            layer_path = os.path.join(st.session_state.datadir, st.session_state.model_name, layer)
            os.makedirs(layer_path, exist_ok=True)
            image.save(os.path.join(layer_path, name) + '.jpg')
        
        # add newly generated image to database
        st.session_state.database = st.session_state.database | {layer: {name: image}}

    return image

@st.experimental_memo
def load_images(
    datadir: str,
    model_name: str,
) -> Dict:

    # init new dictionary    
    add_to_db = dict()
    
    # join paths to get subdir for model
    datadir = os.path.join(datadir, model_name)

    # check if there are any images there
    try:
        layers = next(os.walk(datadir))[1]
    except StopIteration:
        print(f'No existing images found under {datadir}')
        return dict()

    # load the existing images for all layers
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

def update_image_db(
    datadir: str,
    model_name: str = None,
) -> None:
    if model_name is not None:
        add_to_db = load_images(datadir, model_name)
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

@st.experimental_singleton
def init(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_name == 'Other (specify below)':
        raise NotImplementedError()
    else:
        model = getattr(models, model_name)
        model = model(pretrained=True)
        
    model.to(device).eval()

    st.session_state.layer_names = get_model_layers(model)

    return model

def display_database(
    given_layer: Optional[str] = None,
) -> None:

    # check if model is selected
    if st.session_state.model is None:
        print('model is None!')
        return
    
    expanders = dict()
    for layer, layer_features in st.session_state.database.items():
        if given_layer is not None and layer != given_layer:
            continue
        expanders[layer] = st.expander(label=layer, expanded=True)
        image_list = list(layer_features.values()) # TODO make sure they are sorted according to their names
        with expanders[layer]:
            st.image(image_list)   


def generate_layer_features(
    model: torch.nn.Module,
    layer: str,
) -> None:
    
    pass


def generate_batch_images(
    model: torch.nn.Module, 
    identifiers: List[str],
):
    
    # only generate images that are not already in database
    if st.session_state.load_data:
        for i, ident in enumerate(identifiers):
            if check_database(identifier) is not None:
                del identifiers[i]
    
    if len(identifiers) > 0:
        # set up parameterization for batch optimization
        batch_size = len(identifiers)
        param_f = param.image(128, batch_size=batch_size)

        objective = Objectives.sum(channel(ident) for ident in identifiers) # TODO hier weitermachen mit batching
        images = (render_vis(model, ident, show_image=False, thresholds=(64,))[0][0] * 255).astype(np.uint8) # TODO
        image = Image.fromarray(image)
        layer, name = split_identifier(identifier)
        if st.session_state.save_data:
            layer_path = os.path.join(st.session_state.datadir, st.session_state.model_name, layer)
            os.makedirs(layer_path, exist_ok=True)
            image.save(os.path.join(layer_path, name) + '.jpg')
        
        # add newly generated image to database
        st.session_state.database = st.session_state.database | {layer: {name: image}}

    return image