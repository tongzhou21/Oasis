import json
import random
import os

import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

from utils import my_set_page
from utils import session_state_initialization
from utils import sample_data

my_set_page()

st.markdown('# 🔬 Instance Inspection')#
st.write('Local Quality Evaluation')

session_state_initialization('corpora_parent_path', '/data/tongzhou/corpus/')
session_state_initialization('text_key', 'content')
session_state_initialization('truncation', 512)

session_state_initialization('corpus_folder_name_selected', '')

session_state_initialization('eval_sample_data_count', 10000)
session_state_initialization('list_dict_data_page', [])

session_state_initialization('data_count_per_page_prev', 5)
session_state_initialization('data_count_per_page', 5)
session_state_initialization('default_quality', 'Undefined')
session_state_initialization('default_quality_prev', 'Undefined')

session_state_initialization('list_label_human_page', ['Undefined' for _ in range(5)])
session_state_initialization('list_label_human_page_history', [])
session_state_initialization('list_dict_data_human_page_history', [])

session_state_initialization('case_save_type', 'New')
session_state_initialization('case_save_filename', '')

session_state_initialization('list_click_human_page', [False for _ in range(5)])
session_state_initialization('gpt4_eval_corpus_name', st.session_state.corpus_folder_name_selected)
session_state_initialization('gpt4_eval_case_save_name', '')

session_state_initialization('gpt4_eval_truncation_length', st.session_state.truncation)

session_state_initialization('gpt4_apikey', '')
session_state_initialization('gpt4_eval_prompt',
    "Please score the follow text for qualification for large language model's pre-train corpus")
session_state_initialization('gpt4_eval_data_count', 200)
session_state_initialization('gpt4_eval_log_path', 'log/temp.gpt4_eval.txt')


#### sidebar
with st.sidebar:
    #### rule corpus setting
    with st.expander('⚙️ Corpus Settings', expanded=True):
        with st.form(key='page_eval_instance_form_corpus_setting'):
            #### select a dataset
            list_folder_name = os.listdir(st.session_state.corpora_parent_path)
            list_options_corpus = [''] + list_folder_name
            st.session_state.corpus_folder_name_selected = st.selectbox(
                "Corpus Folder",
                key='page_eval_instance_corpus_folder',
                options=list_options_corpus,
                index=list_options_corpus.index(st.session_state.corpus_folder_name_selected),
            )

            #### sample line count
            st.session_state.eval_sample_data_count = \
                st.number_input('Sample Data Count Assessment', value=st.session_state.eval_sample_data_count)

            #### truncation
            st.session_state.truncation = st.number_input('Visualization Truncation',
                                                                 value=st.session_state.truncation)

            #### text_key
            st.session_state.text_key = st.text_input('Text Key', key='text key eval instance', value=st.session_state.text_key)

            submit_button = st.form_submit_button(label='Confirm')
            if submit_button:
                #### read data
                st.session_state.list_dict_data = \
                    sample_data(st.session_state.corpora_parent_path + st.session_state.corpus_folder_name_selected,
                                st.session_state.eval_sample_data_count, read_all=True)
                if len(st.session_state.list_dict_data) > 0:
                    st.session_state.list_dict_data_page = random.sample(st.session_state.list_dict_data, 5)

tab_human, tab_ai = st.tabs(["Human Rating", "LLM Evaluation"])


def refresh_page():
    for idx in range(len(st.session_state.list_label_human_page)):
        if st.session_state.list_label_human_page[idx] != 'Undefined':
            st.session_state.list_dict_data_human_page_history.append(st.session_state.list_dict_data_page[idx])
            st.session_state.list_label_human_page_history.append(st.session_state.list_label_human_page[idx])

    st.session_state.list_dict_data_page = random.sample(st.session_state.list_dict_data,
                                                         st.session_state.data_count_per_page)
    st.session_state.list_click_human_page = [False for _ in range(st.session_state.data_count_per_page)]
    st.session_state.list_label_human_page = [st.session_state.default_quality for _ in
                                              range(st.session_state.data_count_per_page)]

def clean_history():
    #### 重置
    st.session_state.list_click_human_page = [False for _ in range(st.session_state.data_count_per_page)]
    st.session_state.list_label_human_page = [st.session_state.default_quality for _ in
                                              range(st.session_state.data_count_per_page)]
    st.session_state.list_label_human_page_history = []
    st.session_state.list_dict_data_human_page_history = []

#### human
with tab_human:
    #### eval sidebar
    with st.sidebar:
        with st.expander('Inspection Panel', expanded=True):

            st.session_state.data_count_per_page = \
                st.number_input('Instances Per Page', min_value=1, max_value=50, value=st.session_state.data_count_per_page)
            if st.session_state.data_count_per_page != st.session_state.data_count_per_page_prev:
                st.session_state.list_dict_data_page = random.sample(st.session_state.list_dict_data,
                                                                     st.session_state.data_count_per_page)
                st.session_state.list_click_human_page = [False for _ in range(st.session_state.data_count_per_page)]
                st.session_state.list_label_human_page = [st.session_state.default_quality for _ in range(st.session_state.data_count_per_page)]
                st.session_state.data_count_per_page_prev = st.session_state.data_count_per_page

            #### default quality
            list_options_default_quality = ["High", "Undefined", "Low"]
            st.session_state.default_quality = st.radio(
                "Default Quality",
                list_options_default_quality,
                index=list_options_default_quality.index(st.session_state.default_quality),
                horizontal=True,
            )
            if st.session_state.default_quality != st.session_state.default_quality_prev:
                st.session_state.list_label_human_page = [st.session_state.default_quality
                                                          for _ in range(st.session_state.data_count_per_page)]
                st.session_state.default_quality_prev = st.session_state.default_quality
            #### record
            st.write('#### Inspection Record')

            col0, col1 = st.columns(2)
            with col0:
                if st.button(label='Refresh & Confirm'):
                    refresh_page()
            with col1:
                if st.button(label='Clear History'):
                    clean_history()

            col0, col1 = st.columns(2)
            with col0:
                hq_count = st.session_state.list_label_human_page.count('High') + \
                           st.session_state.list_label_human_page_history.count('High')
                col0.metric(label=":green[High Quality]",value=hq_count)
                style_metric_cards(border_left_color='Green')
            with col1:
                lq_count = st.session_state.list_label_human_page.count('Low') + \
                           st.session_state.list_label_human_page_history.count('Low')
                col1.metric(label=":red[Low Quality]", value=lq_count)

            st.write('#### Save Cases')
            list_options_write_case = ["New", 'Append']
            st.session_state.case_save_type = st.radio(
                "Write Cases To File",
                list_options_write_case,
                index=list_options_write_case.index(st.session_state.case_save_type),
                horizontal=True,
            )
            if st.session_state.case_save_type == 'New':
                st.session_state.case_save_filename = \
                    st.text_input('Cases Name', key='case name write new', value=st.session_state.case_save_filename)
            if st.session_state.case_save_type == 'Append':
                list_options_case_save = os.listdir('scripts/assessment/case/')
                list_options_case_save = [case_save.replace('.human.pos', '')
                                          for case_save in list_options_case_save if '.human.pos' in case_save]
                list_options_case_save = [''] + list_options_case_save
                st.session_state.case_save_filename = st.selectbox(
                    "Case Name",
                    key='select exist case name',
                    options=list_options_case_save,
                    index=list_options_case_save.index(st.session_state.case_save_filename),
                )

            submit_button = st.button(label='Export', key='export cases')
            if submit_button:
                refresh_page()

                with open('scripts/assessment/case/' + st.session_state.case_save_filename + '.human.pos',
                          'w' if st.session_state.case_save_type == 'New' else 'a') as f_write_hq, \
                        open('scripts/assessment/case/' + st.session_state.case_save_filename + '.human.neg',
                         'w' if st.session_state.case_save_type == 'New' else 'a') as f_write_lq:
                    for idx, label_human in enumerate(st.session_state.list_label_human_page_history):
                        dict_data_write = st.session_state.list_dict_data_human_page_history[idx]
                        if label_human == 'High': f_write_hq.write(json.dumps(dict_data_write, ensure_ascii=False) + '\n')
                        if label_human == 'Low': f_write_lq.write(json.dumps(dict_data_write, ensure_ascii=False) + '\n')

                clean_history()
                st.info('Success')



    #### show cases
    for idx in range(min(st.session_state.data_count_per_page, len(st.session_state.list_dict_data_page))):
        label_cur = 'Undefined'
        if st.session_state.list_label_human_page[idx] == 'High':
            label_cur = ':green[High Quality]'
        if st.session_state.list_label_human_page[idx] == 'Low':
            label_cur = ':red[Low Quality]'
        with st.expander(label_cur, expanded=not st.session_state.list_click_human_page[idx]):
            col1, col2 = st.columns([1, 0.1])

            with col1:
                st.write({'text': st.session_state.list_dict_data_page[idx]
                                  [st.session_state.text_key][:st.session_state.truncation_length]})

            with col2:
                hq = st.button(label=':green[HQ]', key='bt_high' + str(idx))
                lq = st.button(label=':red[LQ]', key='bt_low' + str(idx))

                if hq:
                    st.session_state.list_label_human_page[idx] = 'High'
                    st.session_state.list_click_human_page[idx] = True
                if lq:
                    st.session_state.list_label_human_page[idx] = 'Low'
                    st.session_state.list_click_human_page[idx] = True


with tab_ai:
    st.session_state.flag_click_human = False
    st.write('### GPT-4 Evaluation')
    with st.form(key='GPT-4 eval form'):
        list_options_corpus = [''] + os.listdir(st.session_state.corpora_parent_path)
        st.session_state.corpus_folder_name_selected = st.selectbox(
            "Corpus Folder for Evaluation",
            key='page_eval_gpt4_corpus_folder',
            options=list_options_corpus,
            index=list_options_corpus.index(st.session_state.gpt4_eval_corpus_name),
        )

        st.session_state.gpt4_eval_case_save_name = st.text_input('Case Save Name',
                                                                  value=st.session_state.gpt4_eval_case_save_name)

        st.session_state.gpt4_eval_truncation_length = \
            st.number_input('Request Truncation', value=st.session_state.gpt4_eval_truncation_length)

        st.session_state.gpt4_apikey = st.text_input('OpenAI API Key', value=st.session_state.gpt4_apikey)

        st.session_state.gpt4_eval_prompt = st.text_input('Evaluate Prompt', value=st.session_state.gpt4_eval_prompt)

        st.session_state.gpt4_eval_data_count = st.number_input('Evaluate Count', value=st.session_state.gpt4_eval_data_count)

        st.session_state.gpt4_eval_log_path = st.text_input('Log Path', value=st.session_state.gpt4_eval_log_path)

        submit_button = st.form_submit_button(label='Run')
        if submit_button:
            cmd = 'python -u run/eval/gpt4_eval.py --corpus_path {:s} --text_key {:s} ' \
                  '--sample_count {:d} --api_key {:s} --prompt {:s} --truncation {:d} --save_path {:s}| tee {:s}'.format(
                st.session_state.corpora_parent_path + st.session_state.corpus_folder_name_selected,
                st.session_state.text_key,
                st.session_state.gpt4_eval_data_count,
                st.session_state.gpt4_apikey,
                st.session_state.gpt4_eval_prompt,
                st.session_state.gpt4_eval_truncation_length,
                'scripts/assessment/case/' + st.session_state.gpt4_eval_case_save_name,
                st.session_state.gpt4_eval_log_path,
            )
            st.info(cmd)
            os.system(cmd)



