import json
import random
import os

from streamlit_ace import st_ace
import streamlit as st
import time
from utils import my_set_page
from utils import session_state_initialization
from utils import sample_data
from utils import load_kenlm_model
import hashlib
import Levenshtein
from streamlit_modal import Modal


my_set_page()

st.title('🖨️ Neural Filter')
st.write('Debiased Neural Filter')

# session_state_initialization
session_state_initialization('corpora_parent_path', '/data/tongzhou/corpus/')
session_state_initialization('text_key', 'content')
session_state_initialization('truncation', 512)
session_state_initialization('corpus_folder_name_selected', '')
session_state_initialization('list_dict_data', [])
session_state_initialization('model_sample_data_count', 300000)
session_state_initialization('kenlm_name', '')
session_state_initialization('kenlm_model', None)
session_state_initialization('kenlm_model_leval', 'char')
session_state_initialization('kenlm_sorted_tuple', [])
session_state_initialization('ppl_samples_file_name', st.session_state.corpus_folder_name_selected + 'ppl_samples')
session_state_initialization('kenlm_case_study_sample_count', 10000)
session_state_initialization('ppl_run_log_path', 'log/run_ppl.temp.txt')
session_state_initialization('neg_quantile', 0.8)
session_state_initialization('predefined_contamination', 'cc-zh.v0')
session_state_initialization('contam_name_save', '')
session_state_initialization('contamination_set', [])
session_state_initialization('flag_single_contam_report', False)
session_state_initialization('sample_data_count_contamination_run', 300000)

session_state_initialization('flag_save_contam', False)
session_state_initialization('list_all_contam_name', [])
session_state_initialization('list_contam_text', [])
session_state_initialization('report_sample_count', 1000)
session_state_initialization('corpus_folder_read_contamination_run',
                             st.session_state.corpora_parent_path + st.session_state.corpus_folder_name_selected + '/')
session_state_initialization('corpus_folder_write_contamination_run',
                             st.session_state.corpora_parent_path + st.session_state.corpus_folder_name_selected + '.contamination/')
session_state_initialization('text_key_contamination_run', st.session_state.text_key)
session_state_initialization('contamination_set_name_contamination_run', '')
session_state_initialization('num_worker_contamination_run', 1)
session_state_initialization('accumulate_contamination_prob_contamination_run', 0.1)
session_state_initialization('max_accumulation_contamination_run', 3)
session_state_initialization('contamination_samples_file_name', st.session_state.corpus_folder_name_selected + 'contamination_samples')
session_state_initialization(
    'dict_source2weight',
    {
        'Positive':{
            'HQ Source': 0.1,
            'Low PPL': 0.6,
            'Positive Case': 0.05,
        },
        'Negative':{
            'Contamination': 0.6,
            'High PPL': 0.2,
            'Negative Case': 0.05,
        }
    })
session_state_initialization('hx_script_file_contamination_run', '')
session_state_initialization('log_path_contamination_run', 'log/contamination.temp.txt')
session_state_initialization('pos_append', 'HQ Source')
session_state_initialization('neg_append', 'Contamination')

session_state_initialization('hq_source_folder_name_build_quality_dataset', '')
session_state_initialization('case_name_build_quality_dataset', '')
session_state_initialization('ppl_name_build_quality_dataset', '')

session_state_initialization('contamination_name_build_quality_dataset','')
session_state_initialization('quality_classification_dataset_name', 'temp.quality_dataset')

session_state_initialization('quality_dataset_name_train', '')
session_state_initialization('model_name_train', 'bert.cc')
session_state_initialization('plm_train', 'hfl/chinese-electra-180g-small-discriminator')
session_state_initialization('gpu_idx_train', 0)
session_state_initialization('truncation_train', 500)
session_state_initialization('epoch_count_train', 16)
session_state_initialization('batch_size_train', 16)
session_state_initialization('lr_train', 3e-4)
session_state_initialization('lr_bert_train', 3e-5)
session_state_initialization('weight_decay_bert_train', 0.01)
session_state_initialization('log_path_train', 'log/temp.bert.txt')

session_state_initialization('corpus_folder_read_neural_run', st.session_state.corpus_folder_name_selected)
session_state_initialization('corpus_folder_write_neural_run', st.session_state.corpus_folder_name_selected + '.neural')
session_state_initialization('filter_model_name_neural_run', '')
session_state_initialization('plm_model_name_neural_run', 'prajjwal1/bert-mini')
session_state_initialization('gpu_idx_neural_run', 0)
session_state_initialization('batch_size_neural_run', 32)
session_state_initialization('quality_threshold', 0.8)
session_state_initialization('log_path_neural_run', 'log/neural_filter.temp.txt')

#### sidebar
with st.sidebar:
    #### model corpus setting
    with st.expander('⚙️ Corpus Settings', expanded=True):
        with st.form(key='page_model_form_corpus_setting'):
            #### select a dataset
            list_folder_name = os.listdir(st.session_state.corpora_parent_path)
            list_options_corpus = [''] + list_folder_name

            st.session_state.corpus_folder_name_selected = st.selectbox(
                "Corpus Folder",
                key='page_model_corpus_folder',
                options=list_options_corpus,
                index=list_options_corpus.index(st.session_state.corpus_folder_name_selected),
            )

            #### sample line count
            st.session_state.model_sample_data_count = \
                st.number_input('Sample Data Count Quality Corpus', value=st.session_state.model_sample_data_count)

            #### truncation
            st.session_state.truncation_length = \
                st.number_input('Compute Truncation', value=st.session_state.truncation)

            #### text_key
            st.session_state.text_key = st.text_input('Text Key', value=st.session_state.text_key)

            #### contamination
            list_contam_name = os.listdir('scripts/neural/')
            list_contam_name = [contam_name.replace('.contamination', '')
                                for contam_name in list_contam_name if '.contamination' in contam_name]

            st.session_state.predefined_contamination = st.selectbox(
                'Predefined Contamination Strategy',
                options=list_contam_name,
                index=list_contam_name.index(st.session_state.predefined_contamination)
            )

            submit_button = st.form_submit_button(label='Confirm')
            if submit_button:
                #### read data
                st.session_state.list_dict_data = \
                    sample_data(st.session_state.corpora_parent_path + st.session_state.corpus_folder_name_selected,
                                st.session_state.model_sample_data_count, read_all=True)
                #### load contamination
                if st.session_state.contamination_set == []:
                    with open('scripts/neural/' + st.session_state.predefined_contamination + '.contamination', 'r') as f_read:
                        predefined_contamination_jsonline = [json.loads(line) for line in f_read]
                    st.session_state.contamination_set = predefined_contamination_jsonline

tab_ppl, tab_contamination, tab_mix, tab_run = st.tabs(["Perplexity", "Contamination", 'Recipe', 'Run Neural'])


def calculate_kenlm_score(list_text, kenlm_model):
    list_score = []
    for idx, text in enumerate(list_text):
        list_score.append(kenlm_model.perplexity(text))

    list_sorted_tuple = sorted([(list_text[i], list_score[i]) for i in range(len(list_text))],
                               key=lambda v: (v[1], v[0]), reverse=False)
    return list_sorted_tuple

with tab_ppl:
    with st.expander('Kenlm Score', expanded=True):
        st.write('#### Kenlm Model')
        # with st.expander('PPL Info', expanded=True):
        with st.form(key='kenlm case study form'):
            list_options_kenlm = [''] + os.listdir('scripts/klm/')
            st.session_state.kenlm_name = st.selectbox(
                "Kenlm Model",
                list_options_kenlm,
                index=list_options_kenlm.index(st.session_state.kenlm_name)
            )
            st.session_state.kenlm_model_leval = st.radio("Kenlm Model Level", ['char', 'word'], horizontal=True)


            st.session_state.kenlm_case_study_sample_count = \
                st.number_input('Case Study Sample Count', value=st.session_state.kenlm_case_study_sample_count)
            submit_button = st.form_submit_button(label='Load Kenlm Model')
            if submit_button:
                st.session_state.kenlm_model = load_kenlm_model(st.session_state.kenlm_name)

        st.write('#### Quantile Analysis')

        if st.button('Run Kenlm For Case Study'):
            list_text = [dict_data[st.session_state.text_key][:st.session_state.truncation]
                         for dict_data in st.session_state.list_dict_data[:st.session_state.kenlm_case_study_sample_count]]
            if st.session_state.kenlm_model_leval == 'char':
                list_text = [' '.join(text) for text in list_text]

            st.session_state.kenlm_sorted_tuple = calculate_kenlm_score(list_text, st.session_state.kenlm_model)

        pos = st.slider('Quantile Inspection', min_value=0.0, max_value=100.0, step=0.1)

        if len(st.session_state.kenlm_sorted_tuple) != 0:
            tuple_ppl = st.session_state.kenlm_sorted_tuple[
                int((len(st.session_state.kenlm_sorted_tuple) - 1) * pos // 100)]
            st.write('#### ppl: {:.4f}'.format(tuple_ppl[1]))
            st.write('#### text')
            st.write({'text': tuple_ppl[0]
            if st.session_state.kenlm_model_leval == 'word' else tuple_ppl[0].replace(' ', '')})


    with st.expander('Split Positive and Negative by Perplexity', expanded=True):
        st.write('#### Run Perplexity')
        with st.form(key='Run Perplexity'):
            st.session_state.neg_quantile = \
                st.number_input('Run Perplexity Negative Quantile', value=st.session_state.neg_quantile, step=0.1)


            st.session_state.ppl_samples_file_name = \
                st.text_input(label='Perplexity Samples File Name', value=st.session_state.ppl_samples_file_name)

            st.session_state.ppl_run_log_path = \
                st.text_input(label='Run Perplexity Log Path', value=st.session_state.ppl_run_log_path)

            submit_button = st.form_submit_button(label='Run')

            if submit_button:
                cmd = 'python -u run/neural/run_kenlm.py ' \
                      '--kenlm_path {:s} --kenlm_model_type {:s} ' \
                      '--corpus_path {:s} --ppl_save_prefix {:s} ' \
                      '--text_key {:s} --sample_count {:s} ' \
                      '--neg_quantile {:s} --truncation {:s} | tee {:s}'. \
                    format('scripts/klm/' + st.session_state.kenlm_name, st.session_state.kenlm_model_leval,
                           st.session_state.corpora_parent_path + st.session_state.corpus_folder_name_selected,
                           'scripts/neural/ppl/' + st.session_state.ppl_samples_file_name,
                           st.session_state.text_key, str(st.session_state.model_sample_data_count),
                           str(st.session_state.neg_quantile), str(st.session_state.truncation),
                           st.session_state.ppl_run_log_path)
                st.info(cmd)
                os.system(cmd)
                st.info('Running Perplexity In The Background')

##### 加载预定义污染策略集合
with open('scripts/neural/contam_set.py', 'r') as f_read:
    for line in f_read:
        if '# Customized Contamination Strategy: ' in line:
            rule_name = line.replace('# Customized Contamination Strategy: ', '').replace('\n', '')
            st.session_state.list_all_contam_name.append(rule_name)

def run_contam_exp(function, args):
    list_hit_text = []
    list_leven_dis_rate = []
    for dict_data in st.session_state.list_dict_data[:st.session_state.report_sample_count]:
        text = dict_data[st.session_state.text_key]
        text_new = function(text, args)

        dis = Levenshtein.distance(text, text_new)
        rate = dis / max(len(text), len(text_new), 1e-9)
        list_leven_dis_rate.append(rate)

        list_hit_text.append((text, text_new))

    st.session_state.list_contam_text = list_hit_text
    return sum(list_leven_dis_rate) / (1e-9 + len(list_leven_dis_rate))


## contamination
with tab_contamination:
    st.write('### Contamination Pool')
    # st.write('#### Add Customize Contamination Strategy')
    with st.expander('Add Customize Contamination Strategy', expanded=False):

        rule_name_custom = st.text_input('Contamination Name')
        if rule_name_custom in st.session_state.list_all_contam_name:
            st.warning('Contamination Name Already Exists!')

        if rule_name_custom and rule_name_custom not in st.session_state.list_all_contam_name:
            template = '###########################\n# Do Not Change The Function Name!\n###########################' \
                       '\n\n# Customized Contamination Strategy: {:s}'.format(rule_name_custom) + \
                       '\ndef contamination(text, args):\n\n    return text'
            content = st_ace(
                template,
                language='python',
                keybinding="sublime",
                theme='twilight',
                font_size=14,
                wrap=True,
                show_gutter=True,
                readonly=False,
            )
            if content and content != '' and content != template:
                hx_name = hashlib.md5(rule_name_custom.encode('utf-8')).hexdigest()
                content = content.replace('def contamination(', 'def md5_{:s}('.format(hx_name))

                with open('scripts/neural/contam_set.py', 'a') as f_write:
                    f_write.write('\n\n' + content + '\n\n')

                st.session_state.list_all_contam_name.append(rule_name_custom)

                st.success('Add {:s} Done!'.format(rule_name_custom))
                content = ''



    st.write('#### Add Contamination Strategy')

    with st.expander('Contamination Pool', expanded=True):
        col0, col1, col2 = st.columns(3)
        with col0: arg0 = st.text_input(label='args[0]', key='args[0] + contam')
        with col1: arg1 = st.text_input(label='args[1]', key='args[1] + contam')
        with col2: contam_weight = st.number_input(label='weight', key='contam_weight', value=1,step=1)

        for contam_name in set(st.session_state.list_all_contam_name):
            if st.button(contam_name, key='contam' + contam_name):
                st.session_state.contamination_set.append({
                    'contam_name': contam_name,
                    'arg0': arg0,
                    'arg1': arg1,
                    'weight': str(contam_weight),
                })
                st.info('add contamination' + contam_name + str((arg0, arg1)))

    st.write('---')
    st.write('### Contamination Set')
    ### setting
    col_pipeline_0, col_pipeline_1, col_pipeline_2 = st.columns(3)

    with col_pipeline_0:
        if st.button('Apply'):
            st.info('Apply Changes!')

    with col_pipeline_1:
        st.session_state.contam_name_save = st.text_input(label='Contamination Set Name',
                                                          value=st.session_state.contam_name_save)

    with col_pipeline_2:
        if st.button('Save Contamination') and st.session_state.contam_name_save != '':
            list_contam_name = os.listdir('scripts/neural/')
            list_contam_name = [contam_name.replace('.contamination', '')
                                for contam_name in list_contam_name if '.contamination' in contam_name]
            list_options_contam = list_contam_name


            def save_neural_contam():
                with open('scripts/neural/{:s}.contamination'.format(st.session_state.contam_name_save),
                          'w') as f_write:
                    for dict_data in st.session_state.contamination_set:
                        f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')
                st.info('Contamination ' + st.session_state.contam_name_save + ' Saved')
                st.session_state.flag_save_contam = False

            if st.session_state.flag_save_contam:
                save_neural_contam()
            else:
                if st.session_state.contam_name_save not in list_options_contam:
                    st.session_state.flag_save_contam = True
                    save_neural_contam()
                else:
                    def warning_callback_contam():
                        st.session_state.flag_save_contam = True

                    with Modal(title='Warning', key='warning modal neural').container():
                        st.write("Overwrite Existing Contamination?")
                        st.write("(If 'Ok', Please Click 'Save Contamination' Again.)")

                        col0, col1 = st.columns(2)
                        with col0: st.button('Ok', key='warning contam', on_click=warning_callback_contam)
                        with col1: st.button('Cancel', key='warning contam')

        if st.button('Clean Contamination'):
            st.session_state.contamination_set = []


    st.write('---')
    #### contamination set
    with st.container():
        for idx, dict_contam in enumerate(st.session_state.contamination_set):
            with st.expander('Strategy: {:d}; Name: {:s}; Weight: {:s}; Args[0]: {:s}; Args[1]: {:s}'.
                format(idx, dict_contam['contam_name'], dict_contam['weight'],
                       dict_contam['arg0'], dict_contam['arg1']), expanded=False):

                tab_args, tab_report = st.tabs(["Args", 'Report'])

                # st.write('##### Setting')
                with tab_args:
                    if st.button('Delete', key='delete' + str(idx)):
                        st.session_state.contamination_set = st.session_state.contamination_set[:idx] + \
                                                             st.session_state.contamination_set[idx + 1:]
                        st.write('please click "Apply" to update page')

                    col0, col1, col2 = st.columns(3)
                    with col0:
                        arg0 = st.text_input(label='args[0]', key='args[0] + contam + pipe' + str(idx), value=dict_contam['arg0'])
                    with col1:
                        arg1 = st.text_input(label='args[1]', key='args[1] + contam + pipe'+ str(idx), value=dict_contam['arg1'])
                    with col2:
                        contam_weight = st.number_input(label='weight', key='contam_weight + pipe'+ str(idx),
                                                        value=int(dict_contam['weight']), step=1)

                    if st.button('Confirm', key='confirm contam pipe' + str(idx)):
                        st.session_state.contamination_set[idx]['arg0'] = arg0
                        st.session_state.contamination_set[idx]['arg1'] = arg1
                        st.session_state.contamination_set[idx]['weight'] = str(contam_weight)


                with tab_report:
                    col0, col1, col2, col3 = st.columns([1, 1, 1, 3])
                    with col0:
                        st.session_state.report_sample_count = \
                            st.number_input('Report Sample', key='report sample contam' + str(idx), value=st.session_state.report_sample_count,step=100)
                    with col1:
                        if st.button('Get Report', key='get report Comtam' + str(idx)):
                            st.session_state.flag_single_contam_report = True
                    with col2:
                        if st.button('Clear Report', key='clear report Comtam' + str(idx)):
                            st.session_state.flag_single_contam_report = False

                    if st.session_state.flag_single_contam_report:
                        hx_name = hashlib.md5(dict_contam['contam_name'].encode('utf-8')).hexdigest()
                        eval("exec('from scripts.neural.contam_set import md5_{:s} as fun')".format(hx_name))

                        edit_rate = run_contam_exp(fun, (dict_contam['arg0'], dict_contam['arg1']))

                        st.write('##### Edit Rate:', edit_rate)

                        if len(st.session_state.list_contam_text) != 0:
                            hit_text_cur = random.choice(st.session_state.list_contam_text)

                            if st.button('Random', key='random show custom' + str(idx)):
                                hit_text_cur = random.choice(st.session_state.list_contam_text)

                            col0, col1 = st.columns(2)
                            with col0:
                                st.write('##### origin')
                                st.write({'text': hit_text_cur[0]})
                            with col1:
                                st.write('##### after')
                                st.write({'text': hit_text_cur[1]})

    st.write('---')
    st.write('### Run Contamination')
    with st.expander('Get Negative by Contamination', expanded=True):
        with st.form(key='corpus_info_form contamination run'):

            st.session_state.corpus_folder_read_contamination_run = \
                st.text_input('Folder Read', key='corpus_folder read contamination run ',
                              value=st.session_state.corpus_folder_read_contamination_run)

            st.session_state.sample_data_count_contamination_run = \
                st.number_input('Sample Data Count',key='sample count contam run', value=st.session_state.sample_data_count_contamination_run)

            st.session_state.contamination_samples_file_name = \
                st.text_input(label='Contamination Samples File Name', value=st.session_state.contamination_samples_file_name)


            st.session_state.text_key_contamination_run = st.text_input('Text Key', key='text key contamination run',
                                                                      value=st.session_state.text_key_contamination_run)

            list_contam_name = os.listdir('scripts/neural/')
            list_contam_name = [contam_name.replace('.contamination', '')
                                for contam_name in list_contam_name if '.contamination' in contam_name]
            list_options_contam = [''] + list_contam_name

            st.session_state.contamination_set_name_contamination_run = st.selectbox(
                "Contamination Set",
                list_options_contam,
                index=list_options_contam.index(st.session_state.contamination_set_name_contamination_run)
            )

            st.session_state.accumulate_contamination_prob_contamination_run = \
                st.number_input('Accumulate Contamination Prob', value=st.session_state.accumulate_contamination_prob_contamination_run)

            st.session_state.max_accumulation_contamination_run = \
                st.number_input('Max Accumulation Count', value=st.session_state.max_accumulation_contamination_run)

            st.session_state.num_worker_contamination_run = \
                st.number_input('Num Workers', key='num workers contam', value=st.session_state.num_worker_contamination_run)

            submit_button = st.form_submit_button(label='Generate Script')

            if submit_button:
                script = ''
                script += 'import sys\nsys.path.insert(0, sys.path[0]+"/../../")\n'
                #### rule pipeline
                with open('scripts/neural/{:s}.contamination'.
                                  format(st.session_state.contamination_set_name_contamination_run), 'r') as f_read:
                    contamination_run = [json.loads(line) for line in f_read]

                list_function_name = ['md5_' + hashlib.md5(dict_rule['contam_name'].encode('utf-8')).hexdigest()
                                      for dict_rule in contamination_run]

                #### import function
                for idx, function_name in enumerate(list_function_name):
                    script_line = 'from scripts.neural.contam_set import {:s} as fun{:d}\n'.format(function_name, idx)
                    script += script_line

                script += 'import random\n'

                script_function = 'def contamination_function(text):\n'

                list_function_str, list_args_str = [], []
                for idx, dict_contam in enumerate(contamination_run):
                    list_function_str += ['fun{:d},'.format(idx) for _ in range(int(dict_contam['weight']))]
                    list_args_str += ['("{:s}", "{:s}"),'.format(dict_contam['arg0'], dict_contam['arg1'])
                                      for _ in range(int(dict_contam['weight']))]
                str_list_function = ' ' * 4 + 'list_function = [' + ' '.join(list_function_str) + ']\n'
                str_list_args = ' ' * 4 + 'list_args = [' + ' '.join(list_args_str) + ']\n'

                script_function += str_list_function
                script_function += str_list_args

                script_function += ' ' * 4 + 'idx_fun = random.randrange(len(list_function))\n'
                script_function += ' ' * 4 + 'text = list_function[idx_fun](text, list_args[idx_fun])\n'

                script_function += ' ' * 4 + 'return text\n'
                script += script_function


                script += 'from run.neural.run_contamination import run_contamination\n'
                script += 'run_contamination(contamination_function, ' \
                          '"{:s}", "{:s}", {:d}, "{:s}", {:d}, {:.2f}, {:d})\n'. \
                    format(st.session_state.corpus_folder_read_contamination_run,
                           st.session_state.text_key_contamination_run,
                           st.session_state.sample_data_count_contamination_run,
                           'scripts/neural/contamination/' + st.session_state.contamination_samples_file_name,
                           st.session_state.num_worker_contamination_run,
                           st.session_state.accumulate_contamination_prob_contamination_run,
                           st.session_state.max_accumulation_contamination_run)

                #### script file name
                hx_script_file_name = 'contamination_md5' + \
                                      hashlib.md5(st.session_state.contamination_set_name_contamination_run.encode(
                                          'utf-8')).hexdigest() + \
                                      '_' + \
                                      time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.py'
                st.session_state.hx_script_file_contamination_run = hx_script_file_name

                with open('run/neural/' + hx_script_file_name, 'w') as f_write:
                    f_write.write(script)

                st.info('script ' + hx_script_file_name + ' generation done!')



    #### run script
    st.write('#### Run Script')
    with st.form(key='run_script_form contamination run'):
        st.session_state.log_path_contamination_run = st.text_input('Log Path', key='log path contam run',
                                                                  value=st.session_state.log_path_contamination_run)
        submit_button = st.form_submit_button(label='Run')

        #### background run
        if submit_button:
            cmd = 'python -u run/neural/{:s} | tee {:s}'.\
                format(st.session_state.hx_script_file_contamination_run, st.session_state.log_path_contamination_run)
            st.info(cmd)
            os.system(cmd)


with tab_mix:
    with st.expander('Mixture Info', expanded=True):

        st.write('### Quality Classification Dataset Management')

        st.write('#### Mixture Management')

        col_pos, col_neg = st.columns(2)

        with col_pos:
            col0, col1 = st.columns(2)
            with col0:
                st.write('##### Positive')
            with col1:
                if st.button('Calibrate Distribution', key='calibrate pos'):
                    total_weight = sum([value for key, value in st.session_state.dict_source2weight['Positive'].items()])
                    for key, value in st.session_state.dict_source2weight['Positive'].items():
                        st.session_state.dict_source2weight['Positive'][key] = value / total_weight

            for key, value in st.session_state.dict_source2weight['Positive'].items():

                st.session_state.dict_source2weight['Positive'][key] = \
                        st.slider(key, min_value=0, max_value=100, step=1, value=int(value * 100)) / 100

            pos_append_options = ['HQ Source', 'Low PPL', 'Positive Case']
            st.session_state.pos_append = \
                st.radio(label="Append Source", key='pos append', options=pos_append_options,
                         index=pos_append_options.index(st.session_state.pos_append), horizontal=True)
        with col_neg:
            col0, col1 = st.columns([0.5,1])
            with col0:
                st.write('##### Negative')
            with col1:
                if st.button('Calibrate Distribution', key='calibrate neg'):
                    total_weight = sum([value for key, value in st.session_state.dict_source2weight['Negative'].items()])
                    for key, value in st.session_state.dict_source2weight['Negative'].items():
                        st.session_state.dict_source2weight['Negative'][key] = value / total_weight

            for key, value in st.session_state.dict_source2weight['Negative'].items():
                st.session_state.dict_source2weight['Negative'][key] = \
                    st.slider(key, min_value=0, max_value=100, step=1, value=int(value * 100)) / 100
            neg_append_options = ['Contamination', 'Low PPL', 'Negative Case']
            st.session_state.neg_append = \
                st.radio(label="Append Source", key='neg append', options=neg_append_options,
                         index=neg_append_options.index(st.session_state.neg_append), horizontal=True)
        ####
        st.write('---')
        st.write('#### Build Dataset')
        with st.form(key='build quality dataset'):
            #### hq source
            list_folder_name = os.listdir(st.session_state.corpora_parent_path)
            list_options_corpus = [''] + list_folder_name
            st.session_state.hq_source_folder_name_build_quality_dataset = st.selectbox(
                "High Quality Source",
                key='hq source',
                options=list_options_corpus,
                index=list_options_corpus.index(st.session_state.hq_source_folder_name_build_quality_dataset),
            )

            #### case source
            list_options_case = os.listdir('scripts/assessment/case/')
            list_options_case = [case_name for case_name in list_options_case if '.pos' in case_name]
            list_options_case = [''] + list_options_case
            st.session_state.case_name_build_quality_dataset = st.selectbox(
                "Case Labeling Source",
                key='case labeling source',
                options=list_options_case,
                index=list_options_case.index(st.session_state.case_name_build_quality_dataset),
            )

            #### ppl source
            list_options_ppl_source = os.listdir('scripts/neural/ppl/')
            list_options_ppl_source = list(set([ppl_source.replace('.pos', '').replace('.neg', '')
                                                for ppl_source in list_options_ppl_source]))
            list_options_ppl_source = [''] + list_options_ppl_source

            st.session_state.ppl_name_build_quality_dataset = st.selectbox(
                "Perplexity Split Source",
                key='ppl split source',
                options=list_options_ppl_source,
                index=list_options_ppl_source.index(st.session_state.ppl_name_build_quality_dataset),
            )

            #### contamination source
            list_options_contam_name = os.listdir('scripts/neural/contamination/')
            list_options_contam_name = list(set([contam_source.replace('.contamination.neg', '')
                                                 for contam_source in list_options_contam_name]))
            list_options_contam_name = [''] + list_options_contam_name

            st.session_state.contamination_name_build_quality_dataset = st.selectbox(
                "Contamination Source",
                key='contamination source qd',
                options=list_options_contam_name,
                index=list_options_ppl_source.index(st.session_state.contamination_name_build_quality_dataset),
            )

            st.session_state.quality_classification_dataset_name = \
                st.text_input('Quality Classification Dataset Name',
                                            value=st.session_state.quality_classification_dataset_name)


            submit_button = st.form_submit_button(label='confirm')
            if submit_button:
                sample_count = st.session_state.model_sample_data_count

                pos_case_data_count = int(sample_count * 0.5 * st.session_state.dict_source2weight['Positive']['Positive Case'])
                hq_data_count = int(sample_count * 0.5 * st.session_state.dict_source2weight['Positive']['HQ Source'])
                low_ppl_data_count = int(sample_count * 0.5 * st.session_state.dict_source2weight['Positive']['Low PPL'])

                neg_case_data_count = int(sample_count * 0.5 * st.session_state.dict_source2weight['Negative']['Negative Case'])
                contamination_data_count = int(sample_count * 0.5 * st.session_state.dict_source2weight['Negative']['Contamination'])
                high_ppl_data_count = int(sample_count * 0.5 * st.session_state.dict_source2weight['Negative']['High PPL'])

                pos_hq_count, pos_ppl_count, pos_case_count, neg_contam_count, neg_ppl_count, neg_case_count = 0,0,0,0,0,0

                list_dict_data_hq, list_dict_data_lq = [], []
                ##### hq source
                list_hq = sample_data(st.session_state.corpora_parent_path +
                                                 st.session_state.hq_source_folder_name_build_quality_dataset, pos_case_data_count)
                pos_hq_count = len(list_hq)

                ##### low_ppl
                with open('scripts/neural/ppl/' + st.session_state.ppl_name_build_quality_dataset + '.pos', 'r') as f_read:
                    list_dict_data_ppl_pos = [json.loads(line) for line in f_read]
                random.shuffle(list_dict_data_ppl_pos)

                list_ppl_pos = list_dict_data_ppl_pos[:hq_data_count]
                pos_ppl_count = len(list_ppl_pos)

                ##### case pos
                with open('scripts/assessment/case/' + st.session_state.case_name_build_quality_dataset + '.pos', 'r') as f_read:
                    list_dict_data_case_pos = [json.loads(line) for line in f_read]
                random.shuffle(list_dict_data_case_pos)
                list_case_pos = list_dict_data_case_pos[:pos_case_data_count]
                pos_case_count = len(list_case_pos)

                ##### mix and append
                list_dict_data_hq = list_hq + list_ppl_pos + list_case_pos
                if len(list_dict_data_hq) < sample_count // 2:
                    append_count = sample_count // 2 - len(list_dict_data_hq)
                    if st.session_state.pos_append == 'HQ Source':
                        list_dict_data_hq += sample_data(st.session_state.corpora_parent_path +
                                                         st.session_state.hq_source_folder_name_build_quality_dataset,
                                                         append_count)
                        pos_hq_count += append_count
                    elif st.session_state.pos_append == 'Low PPL':
                        list_dict_data_hq += list_dict_data_ppl_pos[hq_data_count:][:append_count]
                        pos_ppl_count += append_count
                    else:
                        list_dict_data_hq += list_dict_data_case_pos[pos_case_data_count:][:append_count]
                        pos_case_count += append_count

                ##### case neg
                with open('scripts/assessment/case/' + st.session_state.case_name_build_quality_dataset + '.neg', 'r') as f_read:
                    list_dict_data_case_neg = [json.loads(line) for line in f_read]
                random.shuffle(list_dict_data_case_neg)
                list_case_neg = list_dict_data_case_neg[:neg_case_data_count]
                neg_case_count = len(list_case_neg)

                ##### high ppl
                with open('scripts/neural/ppl/' + st.session_state.ppl_name_build_quality_dataset + '.neg', 'r') as f_read:
                    list_dict_data_ppl_neg = [json.loads(line) for line in f_read]
                random.shuffle(list_dict_data_ppl_neg)
                list_ppl_neg = list_dict_data_ppl_neg[:neg_ppl_count]
                neg_ppl_count += len(list_ppl_neg)

                #### contamin
                with open('scripts/neural/contamination/' +
                          st.session_state.contamination_name_build_quality_dataset + '.contamination.neg', 'r') as f_read:
                    list_dict_data_contamination= [json.loads(line) for line in f_read]
                random.shuffle(list_dict_data_contamination)
                list_contam = list_dict_data_contamination[:contamination_data_count]
                neg_contam_count += len(list_contam)

                ##### mix and append
                list_dict_data_lq = list_case_neg + list_ppl_neg + list_contam
                if len(list_dict_data_lq) < sample_count // 2:
                    append_count = sample_count // 2 - len(list_dict_data_lq)
                    if st.session_state.neg_append == 'Contamination':
                        list_dict_data_lq += list_dict_data_contamination[contamination_data_count:][:append_count]
                        neg_contam_count += append_count
                    elif st.session_state.neg_append == 'High PPL':
                        list_dict_data_lq += list_dict_data_ppl_neg[neg_ppl_count:][:append_count]
                        neg_ppl_count += append_count
                    else:
                        list_dict_data_lq += list_dict_data_case_neg[neg_case_data_count:][:append_count]
                        neg_case_count += append_count

                #### mix pos and neg
                list_dict_write = [{'text':dict_data[st.session_state.text_key], 'label': 1} for dict_data in list_dict_data_hq]
                list_dict_write += [{'text':dict_data[st.session_state.text_key], 'label': 0} for dict_data in list_dict_data_lq]
                random.shuffle(list_dict_write)

                with open('scripts/neural/quality/' + st.session_state.quality_classification_dataset_name, 'w') as f_write:
                    for dict_data in list_dict_write:
                        f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')

                st.info('quality dataset build done!')

    with st.expander('Train Model', expanded=True):
        st.write('#### Train Neural Classifier')
        with st.form(key='Quality Classifier'):
            ## 'scripts/neural/quality/'
            list_quality_dataset_name = os.listdir('scripts/neural/quality/')
            list_quality_dataset_name = [''] + list_quality_dataset_name
            st.session_state.quality_dataset_name_train = st.selectbox(
                "Quality Classification Dataset",
                key='quality data name train',
                options=list_quality_dataset_name,
                index=list_quality_dataset_name.index(st.session_state.quality_dataset_name_train),
            )

            st.session_state.model_name_train = st.text_input('Model Name', value=st.session_state.model_name_train)
            st.session_state.plm_train = st.text_input('Pretrained Language Model', value=st.session_state.plm_train)
            st.session_state.gpu_idx_train = st.number_input('GPU index', value=st.session_state.gpu_idx_train)
            st.session_state.truncation_train = st.number_input('Truncation', value=st.session_state.truncation_train)
            st.session_state.epoch_count_train = st.number_input('Epoch', value=st.session_state.epoch_count_train)
            st.session_state.batch_size_train = st.number_input('Batch Size', value=st.session_state.batch_size_train)

            st.session_state.lr_train = st.number_input('Learning Rate', value=st.session_state.lr_train, step=1e-5, format='%.5f')
            st.session_state.lr_bert_train = st.number_input('Learning Rate PLM', value=st.session_state.lr_bert_train, step=1e-5, format='%.5f')

            st.session_state.weight_decay_bert_train = \
                st.number_input('Weight Decay PLM', value=st.session_state.weight_decay_bert_train, step=1e-5, format='%.5f')

            st.session_state.log_path_train = st.text_input('Log Path', key='log path train neural classifier',
                                                            value=st.session_state.log_path_train)

            submit_button = st.form_submit_button(label='Confirm')

            if submit_button:
                cmd = 'python -u run/neural/train_neural_filter.py ' \
                      '--dataset {:s} --model_name {:s} --plm {:s} ' \
                      '--gpu {:s} --max_len {:d} --epoch {:d} --batch_size {:d} ' \
                      '--lr {:.6f} --lr_bert {:.6f} --weight_decay_bert {:.6f} | tee {:s}'.format(
                    'scripts/neural/quality/' + st.session_state.quality_dataset_name_train,
                    st.session_state.model_name_train, st.session_state.plm_train,
                    str(st.session_state.gpu_idx_train), st.session_state.truncation_train,
                    st.session_state.epoch_count_train, st.session_state.batch_size_train,
                    st.session_state.lr_train, st.session_state.lr_bert_train, st.session_state.weight_decay_bert_train,
                    st.session_state.log_path_train
                )
                st.info(cmd)
                os.system(cmd)


with tab_run:
    st.write('#### Run Neural Filter')

    with st.form(key='Run Neural Filter Setting'):

        #### read
        list_options_corpus = [''] + os.listdir(st.session_state.corpora_parent_path)
        st.session_state.corpus_folder_read_neural_run = st.selectbox(
            "Corpus Folder Read",
            key='corpus folder read neural run',
            options=list_options_corpus,
            index=list_options_corpus.index(st.session_state.corpus_folder_read_neural_run),
        )
        #### write
        st.session_state.corpus_folder_write_neural_run = st.text_input('Corpus Folder Write',
                        key='corpus folder write run neural', value=st.session_state.corpus_folder_write_neural_run)
        #### model
        list_options_model = [''] + os.listdir('scripts/neural/model/')
        st.session_state.filter_model_name_neural_run = st.selectbox(
            'Neural Filter',
            key='model name neural run',
            options=list_options_model,
            index=list_options_model.index(st.session_state.filter_model_name_neural_run),
        )
        #### plm
        st.session_state.plm_model_name_neural_run = \
            st.text_input('Pretrained Language Model', key='plm neural run', value=st.session_state.plm_model_name_neural_run)
        #### gpu
        st.session_state.gpu_idx_neural_run = \
            st.number_input('GPU idx', key='gpu idx neural run', value=st.session_state.gpu_idx_neural_run)
        #### batch
        st.session_state.batch_size_neural_run = \
            st.number_input('Batch Size Inference', key='batch_size_inference', value=st.session_state.batch_size_neural_run)
        #### threshold
        st.session_state.quality_threshold = \
            st.number_input('Quality Threshold', key='quality threshold neural run', value=st.session_state.quality_threshold, step=0.01)

        st.session_state.log_path_neural_run = st.text_input('Log Path', key='log_pth_neural run', value=st.session_state.log_path_neural_run)

        submit_button = st.form_submit_button(label='confirm')

        if submit_button:
            cmd = 'python -u run/neural/run_neural_filter.py ' \
                  '--corpus_read {:s} --corpus_write {:s} ' \
                  '--model {:s} --bert {:s} --gpu {:s} --batch_size {:d} --th {:.2f} | tee {:s}'.\
                format(st.session_state.corpora_parent_path + st.session_state.corpus_folder_read_neural_run,
                       st.session_state.corpora_parent_path + st.session_state.corpus_folder_write_neural_run,
                       st.session_state.filter_model_name_neural_run, st.session_state.plm_model_name_neural_run,
                       str(st.session_state.gpu_idx_neural_run), st.session_state.batch_size_neural_run,
                       st.session_state.quality_threshold, st.session_state.log_path_neural_run)
            st.info(cmd)
            os.system(cmd)







