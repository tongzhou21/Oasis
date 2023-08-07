import sys
sys.path.insert(0, sys.path[0]+"/../../")
import json
import tqdm

import random
random.seed(0)

import kenlm
import argparse

from utils import sample_data_without_cache


def run_kenlm_quality_classification(kenlm_path, kenlm_model_type, corpus_path,
                                     ppl_save_prefix, text_key, sample_count, neg_quantile, truncation):
    kenlm_model = kenlm.Model(kenlm_path)

    list_dict_data = sample_data_without_cache(corpus_path, sample_count)

    list_text = [dict_data[text_key] for dict_data in list_dict_data]
    if kenlm_model_type == 'char':
        list_text = [' '.join(text) for text in list_text]

    list_text = [' '.join(text.split(' ')[:truncation]) for text in list_text]

    list_score = []
    for idx, text in tqdm.tqdm(enumerate(list_text), total=len(list_text), ncols=100):
        list_score.append(kenlm_model.perplexity(text))

    list_sorted_tuple = sorted([(i, list_score[i]) for i in range(len(list_text))],
                               key=lambda v: (v[1], v[0]), reverse=False) # ppl: small -> large

    #### TODO: 使用帕累托分布
    list_pos_idx = [t[0] for t in list_sorted_tuple[:int(len(list_sorted_tuple) * neg_quantile)]]
    list_neg_idx = [t[0] for t in list_sorted_tuple[int(len(list_sorted_tuple) * neg_quantile):]]

    with open(ppl_save_prefix + '.pos', 'w') as f_write:
        for idx in list_pos_idx:
            f_write.write(json.dumps(list_dict_data[idx], ensure_ascii=False) + '\n')

    with open(ppl_save_prefix + '.neg', 'w') as f_write:
        for idx in list_neg_idx:
            f_write.write(json.dumps(list_dict_data[idx], ensure_ascii=False) + '\n')
    print('Run Kenlm Quality Dataset Done!')

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kenlm_path", type=str)
    parser.add_argument("--kenlm_model_type", type=str, default='char')
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--ppl_save_prefix", type=str)


    parser.add_argument("--text_key", type=str, default='content')
    parser.add_argument("--sample_count", type=int, default=300000)
    parser.add_argument("--neg_quantile", type=float, default=0.8)
    parser.add_argument("--truncation", type=int, default=1024)

    args = parser.parse_args()

    for arg in vars(args):
        print('{} = {}'.format(arg.lower(), getattr(args, arg)))
    print('')

    run_kenlm_quality_classification(
        args.kenlm_path,
        args.kenlm_model_type,
        args.corpus_path,
        args.ppl_save_prefix,
        args.text_key,
        args.sample_count,
        args.neg_quantile,
        args.truncation,
    )