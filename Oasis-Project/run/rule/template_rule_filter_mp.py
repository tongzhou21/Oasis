import os
import time
import json
from multiprocessing import Pool

import random
random.seed(0)

from utils import get_subfiles_abs_path

def print_error(value):
    print("MP ERROR: ", value)

def rule_filter_single_file(rule_filter, file_path_read, file_path_write, text_key):
    print('rule_filter_single_file: {:s} => {:s}'.format(file_path_read, file_path_write))

    line_count, write_count, edit_rate_sum = 0, 0, 0
    time_start = time.time()

    if not os.path.exists(os.path.split(file_path_write)[0]):
        os.makedirs(os.path.split(file_path_write)[0])
        print('mkdir', os.path.split(file_path_write)[0])

    with open(file_path_read, 'r') as f_read, open(file_path_write, 'w') as f_write:
        for line in f_read:
            line_count += 1
            dict_data = json.loads(line)

            text = rule_filter(dict_data[text_key])

            if text != '':
                edit_rate_sum += len(text) / len(dict_data[text_key])

                dict_data[text_key] = text

                f_write.write(json.dumps(dict_data, ensure_ascii=False) + '\n')
                write_count += 1

    edit_rate = edit_rate_sum / (write_count + 1e-6)
    print('line_count: {:d}; write_count: {:d}; edit_rate: {:.2f}'.format(line_count, write_count, edit_rate), end=' ')
    print('time spent {:.0f}s'.format(time.time() - time_start))


def rule_filter_corpus_multiprocessing(rule_filter, folder_path_read, folder_path_write, text_key, num_workers):
    list_file_path_read = get_subfiles_abs_path(folder_path_read)
    random.shuffle(list_file_path_read)

    list_file_path_write = [file_path_read.replace(folder_path_read, folder_path_write)
                            for file_path_read in list_file_path_read]

    pool = Pool(num_workers)  #
    for idx, file_path_read in enumerate(list_file_path_read):
        file_path_write = list_file_path_write[idx]
        pool.apply_async(func=rule_filter_single_file,
                         args=(rule_filter,
                               file_path_read,
                               file_path_write,
                               text_key,
                               ),
                         error_callback=print_error)
    pool.close()
    pool.join()
    print('Rule Filter END!')

