import json
import random
import os
import copy

from streamlit_ace import st_ace
import streamlit as st

from utils import my_set_page
from utils import sample_data
from utils import session_state_initialization

import hashlib
from streamlit_modal import Modal
import time

my_set_page()

st.title('📝 Rule Filter')
st.write('Interactive Modular Rule Filter')

#### session_state
session_state_initialization('corpora_parent_path', '/data/tongzhou/corpus/')
session_state_initialization('corpus_folder_name_selected', '')
session_state_initialization('rule_sample_data_count', 10000)
session_state_initialization('truncation', 512)
session_state_initialization('text_key', 'text')
session_state_initialization('predefined_pipeline', 'cc-en.v0')
session_state_initialization('rule_pipeline', [])
session_state_initialization('dict_rule_set', {'Drop': [], 'Edit': []}) # TODO:load

session_state_initialization('pipeline_name_save', '')
session_state_initialization('flag_save_pipeline', False)
session_state_initialization('flag_single_rule_report', False)
session_state_initialization('list_hit_text', [])
session_state_initialization('dict_data_cur', {'text': 'please select a dataset!'})

session_state_initialization('corpus_folder_read_rule_filter_run', '/data/tongzhou/corpus/cc_rule-en/raw-may-jun-2023.subset/')
session_state_initialization('corpus_folder_write_rule_filter_run',
                '/data/tongzhou/corpus/cc_rule-en/raw-may-jun-2023.subset' + '.rule/')
session_state_initialization('text_key_rule_filter_run', 'content')
session_state_initialization('rule_pipeline_name_rule_filter_run', '')
session_state_initialization('rule_pipeline_rule_filter_run', [])
session_state_initialization('num_worker_rule_filter_run', 12)
session_state_initialization('flag_rule_filter_run', False)
session_state_initialization('hx_script_file_rule_filter_run', '')
session_state_initialization('log_path_rule_filter_run', 'log/temp.txt')



with open('scripts/rule/rule_set.py', 'r') as f_read:
    for line in f_read:
        if '# Customized Drop Rule: ' in line:
            rule_name = line.replace('# Customized Drop Rule: ', '').replace('\n', '')
            st.session_state.dict_rule_set['Drop'].append(rule_name)
        if '# Customized Edit Rule: ' in line:
            rule_name = line.replace('# Customized Edit Rule: ', '').replace('\n', '')
            st.session_state.dict_rule_set['Edit'].append(rule_name)

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
                key='page_rule_corpus_folder',
                options=list_options_corpus,
                index=list_options_corpus.index(st.session_state.corpus_folder_name_selected),
            )

            #### sample line count
            st.session_state.rule_sample_data_count = \
                st.number_input('Sample Data Count Rule', value=st.session_state.rule_sample_data_count)

            #### truncation
            st.session_state.truncation_length = st.number_input('Visualization Truncation', value=st.session_state.truncation)

            #### text_key
            st.session_state.text_key = st.text_input('Text Key', value=st.session_state.text_key)

            #### pipeline
            list_pipeline_name = os.listdir('scripts/rule/')
            list_pipeline_name = [pipeline_name.replace('.pipeline', '')
                                  for pipeline_name in list_pipeline_name if '.pipeline' in pipeline_name]
            list_options_pipeline = [''] + list_pipeline_name
            st.session_state.predefined_pipeline = st.selectbox(
                'Predefined Rule Filter Pipeline',
                options=list_pipeline_name,
                index=list_pipeline_name.index(st.session_state.predefined_pipeline)
            )
            submit_button = st.form_submit_button(label='Confirm')

            if submit_button:
                #### read data
                st.session_state.list_dict_data = \
                    sample_data(st.session_state.corpora_parent_path + st.session_state.corpus_folder_name_selected,
                                st.session_state.rule_sample_data_count, read_all=True)
                #### load pipeline
                if st.session_state.rule_pipeline == []:
                    with open('scripts/rule/' + st.session_state.predefined_pipeline + '.pipeline', 'r') as f_read:
                        predefined_pipeline_jsonline = [json.loads(line) for line in f_read]
                    st.session_state.rule_pipeline = predefined_pipeline_jsonline

    #### rule set
    st.write('### Rule Set')

    #### Customized Rule
    with st.expander('Add Customized Rule', expanded=False):
        rule_type = st.radio("Rule Type", ['Drop', 'Edit'], horizontal=True)

        rule_name_custom = st.text_input('Rule Name')
        if rule_name_custom in st.session_state.dict_rule_set[rule_type]:
            st.warning('Rule Name Already Exists!')

        if rule_name_custom and rule_name_custom not in st.session_state.dict_rule_set[rule_type]:
            template = '###########################\n# Do Not Change The Function Name!\n###########################' \
                       '\n\n# Customized {:s} Rule: {:s}'.format(rule_type, rule_name_custom) + \
                       '\ndef rule_filter(text, args):\n\n    return text'

            content = st_ace(
                template,
                language='python',
                theme='twilight',
                font_size=12,
                wrap=True,
                show_gutter=True,
                readonly=False
            )
            if content and content != '' and content != template:
                hx_name = hashlib.md5(rule_name_custom.encode('utf-8')).hexdigest()
                content = content.replace('def rule_filter(', 'def md5_{:s}('.format(hx_name))

                with open('scripts/rule/rule_set.py', 'a') as f_write:
                    f_write.write('\n\n' + content + '\n\n')

                st.session_state.dict_rule_set[rule_type].append(rule_name_custom)

                st.success('Add {:s} Done!'.format(rule_name_custom))
                content = ''

    #### rule pool
    tab_drop, tab_edit = st.tabs(["Drop Rules", "Edit Rules"])

    with tab_drop:
        col0, col1 = st.columns(2)
        with col0: arg0 = st.text_input(label='args[0]')
        with col1: arg1 = st.text_input(label='args[1]')
        for rule_drop in set(st.session_state.dict_rule_set['Drop']):
            if st.button(rule_drop, key='drop' + rule_drop):
                st.session_state.rule_pipeline.append({'rule_name': rule_drop,
                                                       'rule_type': 'drop',
                                                       'arg0': arg0,
                                                       'arg1': arg1})

                st.info('add rule drop' + rule_drop + str((arg0, arg1)))

    with tab_edit:
        col0, col1 = st.columns(2)
        with col0: arg0 = st.text_input(label='args[0]', key='args[0] edit')
        with col1: arg1 = st.text_input(label='args[1]', key='args[1] edit')
        for rule_edit in set(st.session_state.dict_rule_set['Edit']):
            if st.button(rule_edit, key='edit' + rule_edit):
                st.session_state.rule_pipeline.append({'rule_name': rule_edit,
                                                       'rule_type': 'edit',
                                                       'arg0': arg0,
                                                       'arg1': arg1})

                st.info('add rule edit' +  rule_edit +  str((arg0, arg1)))
    st.write('---')

tab_case, tab_pipeline, tab_run = st.tabs(["Case Study", "Build Pipeline", 'Run Pipeline'])

#### show case
with tab_case:
    if st.button('Refresh'):
       st.session_state.dict_data_cur = random.choice(st.session_state.list_dict_data)
    with st.expander('Case Study',expanded=True):
        dict_show = copy.copy(st.session_state.dict_data_cur)
        for key, value in dict_show.items():
            if isinstance(value, str):
                dict_show[key] = value[:st.session_state.truncation]
        st.write(dict_show)

st.write('---')


#### run single rule
def run_rule_filter_exp(filter_function, args):
    hit_count = 0
    list_hit_text = []
    for dict_data in st.session_state.list_dict_data:
        text = dict_data[st.session_state.text_key]
        text_new = filter_function(text, args)
        if text_new != text:
            hit_count += 1
            list_hit_text.append((text, text_new))

    st.session_state.list_hit_text = list_hit_text
    return hit_count / len(st.session_state.list_dict_data)


#### build pipeline
with tab_pipeline:
    col_pipeline_0, col_pipeline_1, col_pipeline_2 = st.columns(3)

    with col_pipeline_0:
        if st.button('Apply'):
            st.info('Apply Changes!')

    with col_pipeline_1:
        st.session_state.pipeline_name_save = st.text_input(label='Rule Pipeline Name',
                                                            value=st.session_state.pipeline_name_save)

    with col_pipeline_2:
        if st.button('Save Pipeline') and st.session_state.pipeline_name_save != '':
            list_pipeline_name = os.listdir('scripts/rule/')
            list_pipeline_name = [pipeline_name.replace('.pipeline', '')
                                  for pipeline_name in list_pipeline_name if '.pipeline' in pipeline_name]
            list_options_pipeline = list_pipeline_name


            def save_rule_pipeline():
                with open('scripts/rule/{:s}.pipeline'.format(st.session_state.pipeline_name_save), 'w') as f_write:
                    for dict_data in st.session_state.rule_pipeline:
                        f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')
                st.info('Pipeline ' + st.session_state.pipeline_name_save + ' Saved')
                st.session_state.flag_save_pipeline = False

            if st.session_state.flag_save_pipeline:
                save_rule_pipeline()
            else:

                if st.session_state.pipeline_name_save not in list_options_pipeline:
                    st.session_state.flag_save_pipeline = True
                    save_rule_pipeline()
                else:
                    def warning_callback():
                        st.session_state.flag_save_pipeline = True

                    with Modal(title='Warning', key='warning modal qer').container():
                        st.write("Overwrite Existing Pipeline?")
                        st.write("(If 'Ok', Please Click 'Save Pipeline' Again.)")

                        col0, col1 = st.columns(2)
                        with col0: st.button('Ok', on_click=warning_callback)
                        with col1: st.button('Cancel')

        if st.button('Clean Pipeline'):
            st.session_state.rule_pipeline = []

    st.write('---')
    st.write('### Rule Pipeline')

    with st.container():
        print(len(st.session_state.rule_pipeline))
        for idx, dict_rule in enumerate(st.session_state.rule_pipeline):
            with st.expander('Rule: {:d}; Type: {:s}; Name: {:s}; Args[0]: {:s}; Args[1]: {:s}'.
                                format(idx, dict_rule['rule_type'], dict_rule['rule_name'],
                                       str(dict_rule['arg0']), str(dict_rule['arg1'])), expanded=False):

                tab_pos, tab_args, tab_report = st.tabs(["Basic", "Args", 'Report'])

                with tab_pos:
                    col0, col1, col2, col3 = st.columns([1,1,1,2])
                    with col0:
                        if st.button('⬆️ Move Up', key='move up rule in pipeline' + str(idx)):
                            temp = st.session_state.rule_pipeline[idx - 1]
                            st.session_state.rule_pipeline[idx - 1] = st.session_state.rule_pipeline[idx]
                            st.session_state.rule_pipeline[idx] = temp
                            st.info('please click "Apply" to move up')
                    with col1:
                        if st.button('⬇️ Move Down', key='move down rule in pipeline' + str(idx)):
                            temp = st.session_state.rule_pipeline[idx + 1]
                            st.session_state.rule_pipeline[idx + 1] = st.session_state.rule_pipeline[idx]
                            st.session_state.rule_pipeline[idx] = temp
                            st.info('please click "Apply" to move down')
                    with col2:
                        if st.button('❌ Delete', key='delete rule in pipeline' + str(idx)) :
                            st.session_state.rule_pipeline = st.session_state.rule_pipeline[:idx] + \
                                                                  st.session_state.rule_pipeline[idx + 1:]
                            st.write('please click "Apply" to update page')


                with tab_args:
                    col0, col1 = st.columns(2)
                    with col0: arg0_new = st.text_input('new arg0', key='new arg0' + str(idx))
                    with col1: arg1_new = st.text_input('new arg1', key='new arg1' + str(idx))

                    if st.button('Confirm', key='confirm args' + str(idx)):

                        st.session_state.rule_pipeline[idx]['arg0'] = arg0_new
                        st.session_state.rule_pipeline[idx]['arg1'] = arg1_new

                with tab_report:
                    col0, col1, col2 = st.columns([1,1,3])

                    with col0:
                        if st.button('Get Report', key='get report pipeline' + str(idx)):
                            st.session_state.flag_single_rule_report = True
                    with col1:
                        if st.button('Clear Report', key='clear report' + str(idx)):
                            st.session_state.flag_single_rule_report = False

                    if st.session_state.flag_single_rule_report:
                        hx_name = hashlib.md5(dict_rule['rule_name'].encode('utf-8')).hexdigest()
                        eval("exec('from scripts.rule.rule_set import md5_{:s} as fun')".format(hx_name))

                        hit_rate = run_rule_filter_exp(fun, (dict_rule['arg0'], dict_rule['arg1']))

                        st.write('### Hit Rate:', hit_rate)

                        if hit_rate != 0:
                            hit_text_cur = random.choice(st.session_state.list_hit_text)

                            if st.button('Random', key='random show' + str(idx)):
                                hit_text_cur = random.choice(st.session_state.list_hit_text)

                            col0, col1 = st.columns(2)
                            with col0:
                                st.write('### origin')
                                st.write({'text': hit_text_cur[0]})
                            with col1:
                                st.write('### after')
                                st.write({'text': hit_text_cur[1]})

with tab_run:
    st.write('### Rule Filter Script Generation')
    with st.form(key='corpus_info_form rule filter run'):
        st.session_state.corpus_folder_read_rule_filter_run = \
            st.text_input('Folder Read', value=st.session_state.corpus_folder_read_rule_filter_run)

        st.session_state.corpus_folder_write_rule_filter_run = \
            st.text_input('Folder Write', value=st.session_state.corpus_folder_write_rule_filter_run)

        st.session_state.text_key_rule_filter_run = st.text_input('Text Key', key='text key rule filter run',
                                                                  value=st.session_state.text_key_rule_filter_run)

        list_pipeline_name = os.listdir('scripts/rule/')
        list_pipeline_name = [pipeline_name.replace('.pipeline', '')
                              for pipeline_name in list_pipeline_name if '.pipeline' in pipeline_name]
        list_options_pipeline = [''] + list_pipeline_name
        st.session_state.rule_pipeline_name_rule_filter_run = st.selectbox(
            "Rule Pipeline",
            list_options_pipeline,
            index=list_options_pipeline.index(st.session_state.rule_pipeline_name_rule_filter_run)
        )

        st.session_state.num_worker_rule_filter_run = \
            st.number_input('Num Workers', value=st.session_state.num_worker_rule_filter_run)

        submit_button = st.form_submit_button(label='Generate Script')

        if submit_button:
            script = ''
            script += 'import sys\nsys.path.insert(0, sys.path[0]+"/../../")\n'
            #### rule pipeline
            with open('scripts/rule/{:s}.pipeline'.
                              format(st.session_state.rule_pipeline_name_rule_filter_run), 'r') as f_read:
                rule_pipeline_run = [json.loads(line) for line in f_read]

            list_function_name = ['md5_' + hashlib.md5(dict_rule['rule_name'].encode('utf-8')).hexdigest()
                                  for dict_rule in rule_pipeline_run]

            #### import function
            for idx, function_name in enumerate(list_function_name):
                script_line = 'from scripts.rule.rule_set import {:s} as fun{:d}\n'.format(function_name, idx)
                script += script_line


            script_function = 'def rule_filter(text):\n'
            for idx, dict_rule in enumerate(rule_pipeline_run):
                script_line = ' ' * 4 + 'text = fun{:d}(text, ("{:s}", "{:s}"))'.\
                    format(idx, dict_rule['arg0'], dict_rule['arg1']) + '\n'
                script_function += script_line
            script_function += ' ' * 4 + 'return text\n'
            script += script_function


            script += 'from run.rule.template_rule_filter_mp import rule_filter_corpus_multiprocessing\n'
            script += 'rule_filter_corpus_multiprocessing(rule_filter, "{:s}", "{:s}", "{:s}", {:d})\n'.\
                format(st.session_state.corpus_folder_read_rule_filter_run,
                       st.session_state.corpus_folder_write_rule_filter_run,
                       st.session_state.text_key_rule_filter_run,
                       st.session_state.num_worker_rule_filter_run)

            #### script file name
            hx_script_file_name = 'rule_filter_md5' + \
                           hashlib.md5(st.session_state.rule_pipeline_name_rule_filter_run.encode('utf-8')).hexdigest() + \
                           '_' + \
                           time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + '.py'
            st.session_state.hx_script_file_rule_filter_run = hx_script_file_name

            with open('run/rule/' + hx_script_file_name, 'w') as f_write:
                f_write.write(script)


            st.info('script ' + hx_script_file_name + ' generation done!')

    #### run script
    st.write('### Run Script')
    with st.form(key='run_script_form rule filter run'):
        st.session_state.log_path_rule_filter_run = st.text_input('Log Path', key='log path filter run',
                                                                  value=st.session_state.log_path_rule_filter_run)
        submit_button = st.form_submit_button(label='Run')

        #### background run
        if submit_button:
            cmd = 'python -u run/rule/{:s} | tee {:s}'.\
                format(st.session_state.hx_script_file_rule_filter_run, st.session_state.log_path_rule_filter_run)
            st.info(cmd)
            os.system(cmd)


