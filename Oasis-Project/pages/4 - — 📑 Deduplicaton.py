import json
import os
import streamlit as st
from utils import my_set_page
from utils import session_state_initialization
from utils import sample_data
from utils import jaccard_dedup_n2
from streamlit_agraph import agraph, Node, Edge, Config
import random

my_set_page()

st.title('📑 Deduplication')
st.write('Adaptative Document Deduplication')

session_state_initialization('corpora_parent_path', '/data/tongzhou/corpus/cc_rule-en/')
session_state_initialization('text_key', 'content')
session_state_initialization('truncation', 512)
session_state_initialization('corpus_folder_name_selected', '')
session_state_initialization('dedup_sample_data_count', 10000)
session_state_initialization('agraph_width', 1000)
session_state_initialization('agraph_height', 500)
session_state_initialization('agraph_max_edge_count', 200)

session_state_initialization('lsh_seed_count', 50)
session_state_initialization('lsh_band_count', 5)
session_state_initialization('lsh_ngram', 3)
session_state_initialization('jaccard_th', 0.8)
session_state_initialization('jaccard_ngram', 3)
session_state_initialization('dedup_list_cand_pair', [])
session_state_initialization('dedup_list_cand_pair_show', [])
session_state_initialization('flag_show_graph', False)

session_state_initialization('lsh_seed_count_dedup_run', 100)
session_state_initialization('lsh_band_count_dedup_run', 20)
session_state_initialization('lsh_ngram_dedup_run', 3)
session_state_initialization('jaccard_th_dedup_run', 0.8)
session_state_initialization('jaccard_ngram_dedup_run', 3)

session_state_initialization('cpu_count_dedup_run', 12)
session_state_initialization('mem_size_dedup_run', 250)
session_state_initialization('disk_size_dedup_run', 1000)

session_state_initialization('corpus_disk_size_dedup_run', 500)
session_state_initialization('available_max_memory_dedup_run', 200)
session_state_initialization('minimun_recall_rate_dedup_run', 0.98)
session_state_initialization('insurance_margin_dedup_run', 0.05)

session_state_initialization('recommend_dedup_setting_seed_cunt', 100)
session_state_initialization('recommend_dedup_setting_band_cunt', 10)
session_state_initialization('recommend_dedup_setting_run_cunt', 10)
session_state_initialization('recommend_dedup_setting_recall', 0.9)


#### sidebar
with st.sidebar:
    #### rule corpus setting
    with st.expander('⚙️ Corpus Settings', expanded=True):
        with st.form(key='page_rule_form_corpus_setting'):

            #### select a dataset
            list_folder_name = os.listdir(st.session_state.corpora_parent_path)
            list_options_corpus = [''] + list_folder_name

            st.session_state.corpus_folder_name_selected = st.selectbox(
                "Corpus Folder",
                key='page_dedup_corpus_folder',
                options=list_options_corpus,
                index=list_options_corpus.index(st.session_state.corpus_folder_name_selected),
            )

            #### sample line count
            st.session_state.dedup_sample_data_count = \
                st.number_input('Sample Data Count Dedup', value=st.session_state.dedup_sample_data_count)

            #### truncation
            st.session_state.truncation_length = st.number_input('Visualization Truncation',
                                                                 value=st.session_state.truncation)

            #### text_key
            st.session_state.text_key = st.text_input('Text Key', key='text key dedup', value=st.session_state.text_key)

            submit_button = st.form_submit_button(label='Confirm')

            if submit_button:
                #### read data
                st.session_state.list_dict_data = \
                    sample_data(st.session_state.corpora_parent_path + st.session_state.corpus_folder_name_selected,
                                st.session_state.dedup_sample_data_count, read_all=True)


tab_case, tab_run = st.tabs(["Dedup Case", "Run Dedup"])


with tab_case:
    with st.expander('dedup case study settings', expanded=True):
        with st.form(key='dedup setting for case study'):
            st.session_state.lsh_seed_count = \
                st.number_input('lsh seed count', value=st.session_state.lsh_seed_count, step=10)

            st.session_state.lsh_band_count = \
                st.number_input('lsh band count', value=st.session_state.lsh_band_count, step=1)

            st.session_state.lsh_ngram = \
                st.number_input('lsh n-gram', value=st.session_state.lsh_ngram, step=1)

            st.session_state.jaccard_th = \
                st.number_input('jaccard threshold', value=st.session_state.jaccard_th, step=0.01)

            st.session_state.jaccard_ngram = \
                st.number_input('jaccard n-gram', value=st.session_state.jaccard_ngram, step=1)

            submit_button = st.form_submit_button(label='Confirm')

            if submit_button:
                st.session_state.dedup_list_cand_pair, st.session_state.dedup_list_cand_score = \
                    jaccard_dedup_n2([dict_data[st.session_state.text_key] for dict_data in
                                      st.session_state.list_dict_data[:int(st.session_state.dedup_sample_data_count)]])
                st.write('Candidate Pair Count', len(st.session_state.dedup_list_cand_pair))


    st.write('---')
    st.write('### Case Study')
    nodes, edges, list_nodes = [], [], []

    st.write('#### Total Candidate Pairs:', len(st.session_state.dedup_list_cand_pair))

    st.write('#### Cluster Graph')
    with st.expander('Graph Visualization Settings', expanded=False):
        with st.form(key='dedup setting for agraph'):
            st.session_state.agraph_width = st.number_input('Graph Width', value=st.session_state.agraph_width)
            st.session_state.agraph_height = st.number_input('Graph Height', value=st.session_state.agraph_height)
            st.session_state.agraph_max_edge_count = st.number_input('Max Edge Show', value=st.session_state.agraph_max_edge_count)

            submit_button = st.form_submit_button(label='Confirm')

    col0, col1, col2 = st.columns([1,1,3])
    with col0:
        if st.button('Show Graph'):
            st.session_state.flag_show_graph = True
    with col1:
        if st.button('Clear Graph'):
            st.session_state.flag_show_graph = False

    if st.session_state.flag_show_graph == True:
        random.shuffle(st.session_state.dedup_list_cand_pair)
        st.session_state.dedup_list_cand_pair_show = st.session_state.dedup_list_cand_pair[:st.session_state.agraph_max_edge_count]

        for idx, cand_pair in enumerate(st.session_state.dedup_list_cand_pair_show):
            node0, node1 = str(cand_pair[0]), str(cand_pair[1])
            if node0 not in list_nodes:
                nodes.append(Node(id=node0, label=node0, size=5))
                list_nodes.append(node0)
            if node1 not in list_nodes:
                nodes.append(Node(id=node1, label=node1, size=5))
                list_nodes.append(node1)

            edges.append(Edge(source=node0, target=node1,
                              label="{:.2f}".format(st.session_state.dedup_list_cand_score[idx])))

        config = Config(width=st.session_state.agraph_width,
                        height=st.session_state.agraph_height,
                        directed=False, physics=True, hierarchical=False)

        return_value = agraph(nodes=nodes, edges=edges, config=config)

        st.write('---')
        if return_value:
            st.write(return_value)
            st.write(st.session_state.list_dict_data[int(return_value)])

def recommend_dedup_setting(corpus_disk_size, max_memory_size,
                            jaccard_th, minimun_recall_rate, insurance_margin=0.1):
    max_band_count = 10 * ((1 - insurance_margin) * max_memory_size) / corpus_disk_size
    max_band_count = int(max_band_count)

    p_recall = 1 - ((1 - (jaccard_th ** 10)) ** max_band_count)

    run_count = 1
    recall = 0
    while True:
        recall = 1 - ((1 - p_recall) ** run_count)
        if recall < minimun_recall_rate:
            run_count += 1
        else:
            break

    seed_count = 10 * max_band_count
    return seed_count, max_band_count, run_count, recall


with tab_run:
    with st.expander('Fuzzy Deduplication Setting Recommendation', expanded=True):
        with st.form(key='dedup setting Recommendation for run'):
            st.session_state.corpus_disk_size_dedup_run = \
                st.number_input('Corpus Disk Size (GB)', key='corpus_disk_size_dedup_run run',
                                value=st.session_state.corpus_disk_size_dedup_run, step=5)

            st.session_state.available_max_memory = \
                st.number_input('Max Memory Available (GB)', key='available_max_memory run',
                                value=st.session_state.available_max_memory_dedup_run, step=5)

            st.session_state.jaccard_th_dedup_run = \
                st.number_input('Jaccard Threshold', key='jaccard_th_dedup_run rec run',
                                value=st.session_state.jaccard_th_dedup_run)

            st.session_state.minimun_recall_rate_dedup_run = \
                st.number_input('Min Recall Rate', key='minimun_recall_rate rec run',
                                value=st.session_state.minimun_recall_rate_dedup_run)
            st.session_state.insurance_margin_dedup_run = \
                st.number_input('Memory Insurance Margin', key='insurance_margin_dedup_run rec run',
                                value=st.session_state.insurance_margin_dedup_run)

            submit_button = st.form_submit_button(label='confirm')
            if submit_button:
                st.session_state.recommend_dedup_setting_seed_cunt, \
                st.session_state.recommend_dedup_setting_band_cunt, \
                st.session_state.recommend_dedup_setting_run_cunt, \
                st.session_state.recommend_dedup_setting_recall = recommend_dedup_setting(
                    st.session_state.corpus_disk_size_dedup_run,
                    st.session_state.available_max_memory_dedup_run,
                    st.session_state.jaccard_th_dedup_run,
                    st.session_state.minimun_recall_rate_dedup_run,
                    st.session_state.insurance_margin_dedup_run,
                )

                st.write('#### Recommend Setting:')
                st.write(' * seed: ', st.session_state.recommend_dedup_setting_seed_cunt)
                st.write(' * band: ', st.session_state.recommend_dedup_setting_band_cunt)
                st.write(' * run times: ', st.session_state.recommend_dedup_setting_run_cunt)
                st.write(' * expect recall : ', st.session_state.recommend_dedup_setting_recall)


    with st.expander('Run Fuzzy Deduplication', expanded=True):
        with st.form(key='dedup setting for run'):
            st.session_state.lsh_seed_count_dedup_run = st.number_input('Lsh Seed Count', key='lsh seed count run',
                                                            value=st.session_state.lsh_seed_count_dedup_run, step=10)

            st.session_state.lsh_band_count_dedup_run = \
                st.number_input('Lsh Band Count', key='lsh band count run', value=st.session_state.lsh_band_count_dedup_run, step=1)

            st.session_state.lsh_ngram_dedup_run = \
                st.number_input('Lsh n-gram', key='lsh n-gram run', value=st.session_state.lsh_ngram_dedup_run, step=1)

            st.session_state.jaccard_th_dedup_run = \
                st.number_input('Jaccard Threshold', key='jaccard threshold run', value=st.session_state.jaccard_th_dedup_run, step=0.01)

            st.session_state.jaccard_ngram_dedup_run = \
                st.number_input('Jaccard n-gram', key='jaccard n-gram run', value=st.session_state.jaccard_ngram_dedup_run, step=1)



            st.session_state.cpu_count_dedup_run = st.number_input('Cpu Count', value=st.session_state.cpu_count_dedup_run, step=1)
            st.session_state.mem_size_dedup_run = st.number_input('Max Memory (GB)', value=st.session_state.mem_size_dedup_run, step=10)
            st.session_state.disk_size_dedup_run = st.number_input('Max Disk (GB)', value=st.session_state.disk_size_dedup_run, step=200)

            submit_button = st.form_submit_button(label='confirm')
            if submit_button:
                pass


# recommend_dedup_setting(1000, 190, 0.8, 0.95, 0.05)
