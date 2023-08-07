import json
import random
random.seed(0)
import os
import streamlit as st
import base64
import kenlm

def my_set_page():
    st.set_page_config(
        page_title='Oasis',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    st.markdown("""
        <style>
        div[data-testid='stSidebarNav'] ul {max-height:none}</style>
        """, unsafe_allow_html=True)

    st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=0, padding_bottom=0,
        ),
        unsafe_allow_html=True,
    )
    def add_logo(png_file):
        def get_base64_of_bin_file(png_file):
            with open(png_file, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        binary_string = get_base64_of_bin_file(png_file)
        st.markdown(
            """
            <style>
                [data-testid="stSidebarNav"] {
                    background-image: url("data:image/png;base64,%s");
                    background-repeat: no-repeat;
                    padding-top: 20px;
                    background-position: 20px 2px;
                    overflow: hidden;
                    background-size: 266px 110px;
                }
            </style>
            """ % (binary_string),
            unsafe_allow_html=True,
        ) # 218px 80px; 273px 100px;
    add_logo("resources/logo_web.png") #  242px 100px;
    st.markdown(
        """
       <style>
       [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 400px;
           max-width: 800px;
       }
       """,
        unsafe_allow_html=True,
    )


def print_mp_error(value):
    print("MP ERROR: ", value)

def get_subfiles_abs_path(corpus_folder_path):
    list_abs_path = []
    for root, dirs, files in os.walk(corpus_folder_path, topdown=False):
        for name in files:
            abs_path = os.path.join(root, name)
            list_abs_path.append(abs_path)
    return list_abs_path


@st.cache_data
def sample_data(folder_path, sample_count=10000, read_all=True):
    list_file_path = get_subfiles_abs_path(folder_path)
    random.shuffle(list_file_path)

    sample_count_raw = int(sample_count * 1.5)

    list_line = []
    idx_data = 0
    for file_path in list_file_path:
        with open(file_path, 'r') as f_read:
            for line in f_read:
                idx_data += 1
                if random.random() < sample_count_raw / idx_data:
                    if len(list_line) >= sample_count_raw:
                        if read_all == False: break
                        idx_insert = random.randrange(len(list_line))
                        list_line[idx_insert] = line
                    else:
                        list_line.append(line)

    random.shuffle(list_line)
    list_line = list_line[:sample_count]
    list_dict_data = [json.loads(line) for line in list_line]
    return list_dict_data


def session_state_initialization(name, init):
    if name not in st.session_state:
        st.session_state[name] = init


@st.cache_data
def load_kenlm_model(kenlm_name):
    return kenlm.Model('scripts/klm/' + kenlm_name)


def sample_data_without_cache(folder_path, sample_count=10000, read_all=True):
    list_file_path = get_subfiles_abs_path(folder_path)
    random.shuffle(list_file_path)
    sample_count_raw = int(sample_count * 1.5)
    list_line = []
    idx_data = 0
    for file_path in list_file_path:
        with open(file_path, 'r') as f_read:
            for line in f_read:
                idx_data += 1
                if random.random() < sample_count_raw / idx_data:
                    if len(list_line) >= sample_count_raw:
                        if read_all == False: break
                        idx_insert = random.randrange(len(list_line))
                        list_line[idx_insert] = line
                    else:
                        list_line.append(line)
    random.shuffle(list_line)
    list_line = list_line[:sample_count]
    list_dict_data = [json.loads(line) for line in list_line]
    return list_dict_data

def shingles(text, ngram):
    return set(text[head:head + ngram] for head in range(0, len(text) - ngram + 1))

def jaccard(set_a, set_b):
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


class JaccardSim:
    def __init__(self, ngram):
        self.ngram = ngram

    def shingles(self, text):
        return set(text[head:head + self.ngram] for head in range(0, len(text) - self.ngram + 1))

    def jaccard(self, set_a, set_b):
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    def sim(self, text0, text1):
        shingles_a = self.shingles(text0)
        shingles_b = self.shingles(text1)
        jaccard_sim = self.jaccard(shingles_a, shingles_b)
        return jaccard_sim


def jaccard_dedup_n2(list_text):
    list_cand_pair = []
    list_score = []
    for i, text0 in enumerate(list_text):
        shingles_a = shingles(text0, st.session_state.jaccard_ngram)
        for j, text1 in enumerate(list_text):
            if i == j: continue
            shingles_b =shingles(text1, st.session_state.jaccard_ngram)

            score = jaccard(shingles_a, shingles_b)

            if score > st.session_state.jaccard_th:
                list_cand_pair.append((i,j))
                list_score.append(score)
    return list_cand_pair, list_score



