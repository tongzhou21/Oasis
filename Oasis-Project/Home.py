'''
streamlit run main.py --server.port 7861
'''
import streamlit as st
from PIL import Image
from utils import my_set_page
from utils import session_state_initialization
import torch

my_set_page()

st.image(Image.open('resources/fig_overall_v3.png'))


#### session_state
session_state_initialization('corpora_parent_path', '/data/tongzhou/corpus/cc_rule-en/')
session_state_initialization('free_memory', 256)
session_state_initialization('cpu_count', 12)
session_state_initialization('gpu_count', torch.cuda.device_count())
session_state_initialization('available_gpus', [str(_) for _ in range(st.session_state.gpu_count)])


#### sidebar
with st.sidebar:
    #### system settings
    st.write('### System Settings')
    with st.form(key='page_home_form_system_setting'):

        #### Corpora Parent Path
        st.session_state.corpora_parent_path = \
            st.text_input('Corpora Parent Path', value=st.session_state.corpora_parent_path)

        #### CPU
        st.session_state.cpu_count = \
            st.number_input('Cpu Count', value=st.session_state.cpu_count)

        #### Memory
        st.session_state.free_memory = \
            st.number_input('Free Memory', value=st.session_state.free_memory)

        #### GPU
        st.session_state.available_gpus = st.multiselect(
            label='Available GPUs',
            options=[str(_) for _ in range(st.session_state.gpu_count)],
            default=st.session_state.available_gpus,
        )

        submit_button = st.form_submit_button(label='Confirm System Settings')





