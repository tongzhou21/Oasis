import sys
sys.path.insert(0, sys.path[0]+"/../../")
import json
import tqdm

import random
random.seed(0)

import kenlm
import argparse
import copy

from utils import sample_data_without_cache

'''
def rule_filter_corpus_multiprocessing(rule_filter, folder_path_read, folder_path_write, text_key, num_workers):

'''
def run_contamination(contamination_function,
                      folder_path_read,
                      text_key,
                      sample_count,
                      contamination_save_prefix,
                      num_workers,
                      accumulate_contamination_prob,
                      max_accumulation):

    list_dict_data = sample_data_without_cache(folder_path_read, sample_count)
    list_dict_data_write = []
    for idx, dict_data in tqdm.tqdm(enumerate(list_dict_data), ncols=100, total=len(list_dict_data)):
        text = dict_data[text_key]
        text_new = contamination_function(text)
        for i in range(max_accumulation):
            if random.random() > accumulate_contamination_prob: break
            text_new = contamination_function(text_new)

        dict_data_new = copy.copy(dict_data)
        dict_data_new[text_key] = text_new
        list_dict_data_write.append(dict_data_new)

    with open(contamination_save_prefix + '.contamination.neg', 'w') as f_write:
        for dict_data_write in list_dict_data_write:
            f_write.write(json.dumps(dict_data_write, ensure_ascii=False) + '\n')
    print('Run Contamination Done!')
    return True


'''
run_contamination(contamination_function, "/data/tongzhou/corpus/may-jun-2023.high_quality/may-juu
n-2023.sub04.zh.split.clean/", "/data/tongzhou/corpus/may-jun-2023.high_quality/may-jun-2023.sub00
4.zh.split.clean.contamination/", "text", 1)

'''


