import os
import copy
import streamlit as st
import pyecharts.options as opts # ⚠️注意安装指定版本pip install pyecharts==1.7.0
from pyecharts.charts import Bar
import json
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Line, Liquid, Page, Pie
from pyecharts.commons.utils import JsCode
from pyecharts.components import Table
from pyecharts.faker import Faker
import tqdm
import time
from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts

from utils import my_set_page
from utils import session_state_initialization

my_set_page()

st.markdown('# 📊 Metrics Visualization')#
st.write('Global Distribution Assessment')

session_state_initialization('corpora_parent_path', '/data/tongzhou/corpus/')
session_state_initialization('corpus_folder_name_selected', '')

session_state_initialization('eval_distribution_sample_data_count', 10000)
session_state_initialization(
    'list_heuristic_type', [
        'Lexical Diversity',
        'Semantic Diversity',
        'Task2Vec Diversity',
        'Knowledge Diversity',
        'Similarity to Wikipedia',
        'Topic Diversity',
    ])
session_state_initialization(
    'list_heuristic_type_selected', [
        'Lexical Diversity',
        'Semantic Diversity',
        'Task2Vec Diversity',
        'Knowledge Diversity',
        'Similarity to Wikipedia',
        'Topic Diversity',
    ])
session_state_initialization('list_heuristic_dataset_name', [])
session_state_initialization('heuristic_run_log_path', 'log/temp.heuristic_run.txt')

session_state_initialization('flag_plot', False)
session_state_initialization('list_heuristic_info_name', [])
session_state_initialization('list_heuristic_info', [])

session_state_initialization('list_heuristic_dict_info', [])
session_state_initialization('list_heuristic_dict_info_table', [])
session_state_initialization('dict_data_key2bin', {})
session_state_initialization('list_bar', [])
session_state_initialization('list_bar_key', [])

tab_dis, tab_report = st.tabs(['Heuristic Calculation', 'Report'])
with tab_dis:
    with st.form(key='Heuristic eval form'):
        st.session_state.list_heuristic_type_selected = st.multiselect(
            label='Heuristic Evaluation Function Selection',
            options=st.session_state.list_heuristic_type,
            default=st.session_state.list_heuristic_type_selected,
        )
        list_options_corpus = os.listdir(st.session_state.corpora_parent_path)
        st.session_state.list_heuristic_dataset_name = st.multiselect(
            label='Corpus Selection',
            options=list_options_corpus,
            default=st.session_state.list_heuristic_dataset_name,
            # label_visibility='visible',
            # label_visibility='collapsed',
        )
        st.session_state.heuristic_run_log_path = \
            st.text_input('Log Path', key='heuristic_run_log_path run',value=st.session_state.heuristic_run_log_path)

        submit_button = st.form_submit_button(label='Confirm')
        if submit_button:
            cmd = 'python -u run/eval/gpt4_eval.py | tee {:s}'.format(st.session_state.gpt4_eval_log_path)
            st.info(cmd)

with tab_report:
    with st.sidebar:
        st.write('#### Quality Assessment Comparison')
        with st.form(key='Heuristic eval show info form'):
            list_options_info = os.listdir('scripts/assessment/info/')
            st.session_state.list_heuristic_info_name = st.multiselect(
                label='Overlay Report',
                options=list_options_info,
                default=st.session_state.list_heuristic_info_name,
            )
            submit_button = st.form_submit_button(label='Confirm')
            if submit_button:
                st.session_state.flag_plot = True

    if st.session_state.flag_plot:
        st.session_state.list_heuristic_dict_info = []
        st.session_state.list_heuristic_dict_info_table = []
        for heuristic_info_name in st.session_state.list_heuristic_info_name:
            with open('scripts/assessment/info/' + heuristic_info_name, 'r') as f_read:
                dict_info_list = json.load(f_read)
                dict_info_value = copy.copy(dict_info_list)
                list_key_value, list_key_list = [], []
                for key, value in dict_info_list.items():
                    if not isinstance(value, list):
                        list_key_value.append(key)
                    else:
                        list_key_list.append(key)

                for key in list_key_value:
                    del dict_info_list[key]

                for key in list_key_list:
                    del dict_info_value[key]

                st.session_state.list_heuristic_dict_info.append(dict_info_list)
                st.session_state.list_heuristic_dict_info_table.append(dict_info_value)

        dict_info_table_write = {}
        dict_info_table_write['corpus'] = st.session_state.list_heuristic_info_name

        for key, _ in st.session_state.list_heuristic_dict_info_table[0].items():
            list_value = [dict_data[key] for dict_data in st.session_state.list_heuristic_dict_info_table]
            dict_info_table_write[key] = list_value

        st.write('#### Table')
        st.table(dict_info_table_write)

        #### 分箱信息
        for key, _ in st.session_state.list_heuristic_dict_info[0].items():
            list_list_value = [dict_data[key] for dict_data in st.session_state.list_heuristic_dict_info]
            if key == 'list_kenlm_ppl':
                import math
                for i in range(len(list_list_value)):
                    list_list_value[i] = [math.log(value) for value in list_list_value[i]]

            min_value = min([min(list_value) for list_value in list_list_value])
            max_value = max([max(list_value) for list_value in list_list_value])

            bins = np.linspace(min_value, max_value, 100)

            list_list_bin_count = [[] for _ in range(len(list_list_value))]
            for i in range(len(list_list_value)):

                list_bin_idx = np.digitize(np.array(list_list_value[i]), bins)
                for idx_bin in range(1, 100 + 1):
                    list_list_bin_count[i].append(np.sum(list_bin_idx == idx_bin))

            st.session_state.dict_data_key2bin[key] = {}
            for i in range(len(list_list_value)):
                st.session_state.dict_data_key2bin[key][str(i)] = [int(v) for v in list_list_bin_count[i]]

            st.session_state.dict_data_key2bin[key]['bins'] = [float(v) for v in list(bins)]
        print("allocate bins done")

        dict_key2metric = {
            'list_cluster_size': 'Topic Size',
            'list_doc_length': 'Document Length',
            'list_para_count': 'Paragraph Count',
            'list_mtld_score': 'Lexical Diversity',
            'list_sim_knowledge': 'Knowledge Diversity',
            'list_kenlm_ppl': 'Similarity to Wikipedia',
            'list_sim_semantic': 'Semantic Diversity',
            'list_sim_cluster': 'Topic Diversity',
        }
        #### 绘图
        st.write('#### Charts')

        list_bar = []
        list_key = []
        for key, dict_data in st.session_state.dict_data_key2bin.items():
            category = ['{:.4f}'.format(x) for x in dict_data['bins']]
            corpus_count = len(st.session_state.list_heuristic_info_name)

            list_bar_local = [dict_data[str(i)] for i in range(corpus_count)]

            if key == 'list_doc_length':
                for idx, bar in enumerate(list_bar_local):
                    list_bar_local[idx] = list_bar_local[idx][:-1]


            cur_bar = Bar().add_xaxis(xaxis_data=category)
            for idx_corpus in range(corpus_count):
                cur_bar = cur_bar.add_yaxis(
                    series_name=st.session_state.list_heuristic_info_name[idx_corpus],
                    y_axis=list_bar_local[idx_corpus], label_opts=opts.LabelOpts(is_show=False)
                )
            bar = (cur_bar.set_global_opts(
                    title_opts=opts.TitleOpts(title=''),
                    xaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=False)),
                    yaxis_opts=opts.AxisOpts(
                        axistick_opts=opts.AxisTickOpts(is_show=True),
                        splitline_opts=opts.SplitLineOpts(is_show=True),
                    ),
                )
            )
            st.write(' ')
            list_key.append(dict_key2metric[key])
            list_bar.append(bar)
            st.session_state.list_bar = list_bar
            st.session_state.list_bar_key = list_key

    for idx, bar_cur in enumerate(st.session_state.list_bar):
        st.write('##### {:s}'.format(st.session_state.list_bar_key[idx]))
        st_pyecharts(bar_cur)




