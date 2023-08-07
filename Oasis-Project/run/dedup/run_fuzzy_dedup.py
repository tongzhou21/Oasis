import sys
sys.path.insert(0, sys.path[0]+"/../../")
from lsh import minhash # https://github.com/mattilyra/lsh # ⚠️setup.py USE_CYTHON = True
import os
import time
import tqdm
import gc
import json
from collections import defaultdict
import itertools
import multiprocessing
import argparse

import numpy as np
from multiprocessing import Pool

from utils import get_subfiles_abs_path
from utils import print_mp_error
from utils import shingles
from utils import jaccard


class Cache(object):
    def __init__(self, hasher, num_bands=10):
        self.bins = [defaultdict(set) for _ in range(num_bands)]
        self.hasher = hasher
        if self.hasher:
            msg = 'The number of seeds in the fingerprint must ' \
                  'be divisible by the number of bands'
            assert hasher.num_seeds % num_bands == 0, msg
            self.band_width = hasher.num_seeds // num_bands
        self.num_bands = num_bands

    def bins_(self, fingerprint):
        yield from enumerate(np.array_split(fingerprint, self.num_bands))

    def add_fingerprint(self, fingerprint, doc_id):
        for bin_i, bucket in self.bins_(fingerprint):
            bucket_id = hash(tuple(bucket))
            self.bins[bin_i][bucket_id].add(doc_id)


    def merge_all_bins(self, list_share_bins):
        print('merge_all_bins, len(list_share_bins)', len(list_share_bins))
        bins = [defaultdict(set) for _ in range(self.num_bands)]
        for idx_share_bins, bin in tqdm.tqdm(enumerate(list_share_bins), ncols=50, total=len(list_share_bins)):
            for idx_bin in range(len(bin)):
                for key, set_doc_idx in bin[idx_bin].items():
                    bins[idx_bin][key].update(set_doc_idx)
            list_share_bins[idx_share_bins] = None
        return bins


def get_corpus_files_index(corpus_path_read):
    print('get_corpus_files_index', corpus_path_read)
    list_file_path = get_subfiles_abs_path(corpus_path_read)

    list_file_line_count = []
    total_line_count = 0

    for idx_file, file_path in tqdm.tqdm(enumerate(list_file_path), total=len(list_file_path), ncols=100):
        line_count = 0
        with open(file_path, 'r') as f_read:
            for _ in f_read:
                total_line_count += 1
                line_count += 1
        list_file_line_count.append(line_count)

    #### knapsack problem
    list_file_line_count, list_file_path = \
        (list(t) for t in zip(*sorted(zip(list_file_line_count, list_file_path), reverse=True)))
    list_file_line_count_cumsum = np.cumsum(np.array(list_file_line_count))

    list_init_line_idx = [0] + list(list_file_line_count_cumsum[:-1])
    print('get_corpus_files_index done', 'total line', total_line_count, 'total file count', len(list_file_path))

    return list_file_path, list_init_line_idx, total_line_count


def lsh_cache_single_file(file_path_read, init_line_idx=0, text_key='text',
                          char_ngram=3, seeds=100, bands=20, hashbytes=4, list_share_bin=None):
    hasher = minhash.MinHasher(seeds=seeds, char_ngram=char_ngram, hashbytes=hashbytes)
    if seeds % bands != 0:
        raise ValueError('Seeds has to be a multiple of bands. {} % {} != 0'.format(seeds, bands))

    lsh_cache = Cache(num_bands=bands, hasher=hasher)

    line_idx = init_line_idx
    with open(file_path_read, 'r') as f_read:
        for line in f_read:
            dict_data = json.loads(line)
            text = dict_data[text_key]

            fingerprint = hasher.fingerprint(text.encode('utf8'))
            lsh_cache.add_fingerprint(fingerprint, doc_id=line_idx)

            line_idx += 1

    list_share_bin.append(lsh_cache.bins)

    gc.collect()


def lsh_cache_corpus_mp(list_file_path, list_init_line_idx, text_key,
                 char_ngram=3, seeds=100, bands=20, hashbytes=4, num_workers=24):
    print('lsh_cache_corpus_mp', 'seeds', seeds, 'bands', bands, 'workers', num_workers)
    manager = multiprocessing.Manager()
    list_share_bin = manager.list()

    pbar = tqdm.tqdm(total=len(list_file_path), ncols=100)
    pbar.set_description('lsh_cache_mp')
    update_tqdm = lambda *args: pbar.update()

    pool = Pool(num_workers)
    for idx, file_path in enumerate(list_file_path):
        pool.apply_async(func=lsh_cache_single_file,
                         args=(file_path,
                               list_init_line_idx[idx],
                               text_key,
                               char_ngram, seeds, bands, hashbytes,
                               list_share_bin,),
                         error_callback=print_mp_error,
                         callback=update_tqdm)
    pool.close()
    pool.join()
    gc.collect()
    print('lsh_cache_corpus_mp DONE!')
    return list_share_bin


def bins2candidates(list_share_bins, bands=20):
    print('bins2candidates, len(list_share_bins)', len(list_share_bins), 'bands', bands)

    lsh_cache = Cache(hasher=None, num_bands=bands)
    bins = lsh_cache.merge_all_bins(list_share_bins)

    gc.collect()

    candidate_pairs = set()
    for b in tqdm.tqdm(bins, ncols=100, desc='bins2candidates, get cand'):
        for bucket_id in b:
            if len(b[bucket_id]) > 1:
                pairs_ = set(itertools.combinations(b[bucket_id], r=2))
                candidate_pairs.update(pairs_)
    print('bins2candidates len(candidate_pairs)', len(candidate_pairs))

    gc.collect()
    return candidate_pairs


def read_candidate_text(list_file_path, text_key):
    print('read_candidate_text', end='')
    set_dup_all = set([value for cand in candidates for value in cand])
    print('len(set_dup_all)', len(set_dup_all))
    line_idx = 0
    list_raw_text = []
    for idx, file_path in tqdm.tqdm(enumerate(list_file_path), ncols=100, total=len(list_file_path)):
        with open(file_path, 'r') as f_read:
            for line in f_read:
                dict_data = json.loads(line)
                if line_idx in set_dup_all:
                    text = dict_data[text_key]
                else: text = None
                list_raw_text.append(text)
                line_idx += 1
    return list_raw_text

def calculate_jaccard_score_imap(cand_pair):
    text0, text1 = cand_pair[0], cand_pair[1]

    shingles_a = shingles(text0, 5)  # TODO: n-gram
    shingles_b = shingles(text1, 5)
    jaccard_sim = jaccard(shingles_a, shingles_b)
    return jaccard_sim


def calculate_real_jaccard_candidates_mp(candidates, list_raw_text, th=0.80, num_workers=24):
    np_cand = np.array(list(candidates))

    chunk_size = min(200000, len(candidates) // num_workers)
    chunk_size = min(chunk_size, len(candidates) // num_workers)
    if chunk_size == 0: chunk_size = 1

    print('calculate_real_jaccard_candidates_mp', 'ngram_jaccard', ngram_jaccard, 'th', th)

    def generate_mission_list(list_cand_pair, list_raw_text):
        for cand_pair in list_cand_pair:
            yield (list_raw_text[cand_pair[0]], list_raw_text[cand_pair[1]])


    list_jaccard_sim = []
    mission_list = generate_mission_list(np_cand, list_raw_text)
    with Pool(num_workers) as pool:
        for res in tqdm.tqdm(pool.imap(calculate_jaccard_score_imap, mission_list, chunksize=chunk_size),
                             total=len(np_cand), ncols=100, desc='jaccard_sim_mp'):
            list_jaccard_sim.append(res)

    list_final_pair = [pair for idx, pair in enumerate(np_cand) if list_jaccard_sim[idx] > th]
    list_final_sim = [sim for sim in list_jaccard_sim if sim > th]
    print('calculate_real_jaccard_candidates_mp, list_final_pair', len(list_final_pair))
    return list_final_pair, list_final_sim


def write_result(list_file_path, list_final_pair,
                 corpus_path_read, corpus_path_write, file_path_dup):
    print('write_result', corpus_path_write,  file_path_dup)
    set_dup_remove = set([])
    for pair in tqdm.tqdm(list_final_pair, ncols=50, total=len(list_final_pair)):
        idx0, idx1 = pair[0], pair[1]
        if idx0 not in set_dup_remove and idx1 not in set_dup_remove:
            set_dup_remove.add(idx0)
    print('write_result len(set_dup_remove)', len(set_dup_remove))

    write_line_count = 0
    line_idx = 0
    list_line_write_remove = []
    for idx, file_path in tqdm.tqdm(enumerate(list_file_path), ncols=100, total=len(list_file_path), desc='write dedup'):
        file_path_write = file_path.replace(corpus_path_read, corpus_path_write)
        if not os.path.exists(os.path.split(file_path_write)[0]):
            os.makedirs(os.path.split(file_path_write)[0])
        with open(file_path, 'r') as f_read, open(file_path_write, 'w') as f_write:
            for line in f_read:
                if line_idx not in set_dup_remove:
                    write_line_count += 1
                    f_write.write(line)
                else:
                    list_line_write_remove.append(line)
                line_idx += 1

    if not os.path.exists(os.path.split(file_path_dup)[0]):
        os.makedirs(os.path.split(file_path_dup)[0])
    with open(file_path_dup, 'w') as f_write:
        for line_write in list_line_write_remove:
            f_write.write(line_write)

    return write_line_count / (line_idx + 1)


def recall_prob(jaccard_th, seed_count, band_count):
    return 1 - (1 - jaccard_th ** (seed_count // band_count)) ** band_count

if __name__ == '__main__':
    time_start = time.time()

    parser = argparse.ArgumentParser()

    ##### ⚙️config #####
    corpus_path_read = '/data/tongzhou/corpus/may-jun-2023.high_quality/'
    corpus_path_write = '/data/tongzhou/corpus/may-jun-2023.high_quality.dedup/'
    file_path_dup = '/data/tongzhou/corpus/may-jun-2023.high_quality.dup.jsonl'
    file_path_dup_pair = '/data/tongzhou/corpus/may-jun-2023.high_quality.dup.pair.jsonl'

    text_key = 'content'
    ngram_hasher = 5
    ngram_jaccard = 5 #

    lsh_seed_count = 200 # 150
    lsh_band_count = 10 # 15
    th_jaccard = 0.8

    num_workers = 24

    ##### ⚙️config #####

    multiprocessing.set_start_method('spawn')

    #### index line
    list_file_path, list_init_line_idx, total_line_count = get_corpus_files_index(corpus_path_read)

    #### fingerprint
    list_share_bin = lsh_cache_corpus_mp(
        list_file_path=list_file_path, list_init_line_idx=list_init_line_idx, text_key=text_key,
        char_ngram=ngram_hasher, seeds=lsh_seed_count, bands=lsh_band_count, hashbytes=4, num_workers=num_workers)

    #### candidates
    candidates = bins2candidates(list_share_bins=list_share_bin, bands=lsh_band_count)

    #### read all cand doc
    list_raw_text = read_candidate_text(list_file_path, text_key)

    #### real jaccard
    list_final_pair, list_final_sim = calculate_real_jaccard_candidates_mp(candidates, list_raw_text,
                                                                           th=th_jaccard, num_workers=num_workers)

    with open(file_path_dup_pair, 'w') as f_write:
        for idx, pair in enumerate(list_final_pair):
            text0, text1 = list_raw_text[pair[0]], list_raw_text[pair[1]]
            dict_write = {
                'text0': text0,
                'text1': text1,
                'pair': [int(pair[0]), int(pair[1])],
                'sim': list_final_sim[idx],
            }
            f_write.write(json.dumps(dict_write, ensure_ascii=False) + '\n')

    #### write res
    write_result(list_file_path, list_final_pair, corpus_path_read, corpus_path_write, file_path_dup)

    print('ALL DONE', time.time() - time_start)


