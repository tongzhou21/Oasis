import sys
sys.path.insert(0, sys.path[0]+"/../../")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # TODO: 设定gpu
from transformers import AutoModel, BertModel
import time
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import tqdm
from flashtext import KeywordProcessor
import jieba
from sklearn.metrics.pairwise import cosine_similarity
import kenlm
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, GPT2Tokenizer, default_data_collator

import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from utils import sample_data_without_cache
from lexicalrichness import LexicalRichness

from utils_task2vec import DatasetTask2Vec, montecarlo_fisher, extract_embedding,\
    pdist, stats_of_distance_matrix, sample_task2vec_corpus


class TextDataset(Dataset):
    def __init__(self, plm_name, list_text, max_text_len=500):
        self.tokenizer = AutoTokenizer.from_pretrained(plm_name)  # TODO:
        self.list_text = list_text
        self.max_text_len = max_text_len

    def str2bert_input(self, text, max_length):
        input_ids = self.tokenizer(text)['input_ids']

        input_ids = input_ids[:max_length]
        padding_mask = [1] * len(input_ids)

        input_ids += [0] * (max_length - len(input_ids))
        padding_mask += [0] * (max_length - len(padding_mask))

        return input_ids, padding_mask

    def __len__(self):
        return len(self.list_text)

    def __getitem__(self, item):
        str_text = self.list_text[item]
        input_ids, padding_mask = self.str2bert_input(str_text, self.max_text_len)

        dict_input = {
            'input_ids': input_ids,
            'padding_mask': padding_mask,
            'item': item,
        }
        dict_input = self.get_tensor_dict(dict_input)
        return dict_input

    def get_tensor_dict(self, dict_data):
        for key, value in dict_data.items():
            if isinstance(value, dict):
                dict_data[key] = self.get_tensor_dict(value)
            else:
                dict_data[key] = torch.tensor(value)
        return dict_data

class SentenceVec(object):
    def __init__(self, plm_path='hfl/chinese-electra-180g-small-discriminator', device='cuda'):
        self.plm_path = plm_path
        self.device = device
        if 'chinese-roberta-wwm-ext' in plm_path:
            self.model = BertModel.from_pretrained(plm_path).to(device)
        else:
            self.model = AutoModel.from_pretrained(plm_path).to(device)
        self.model.eval()

    def dict_data2gpu(self, dict_data):
        for key, value in dict_data.items():
            if isinstance(value, dict):
                dict_data[key] = self.dict_data2gpu(value)
            else:
                dict_data[key] = value.to(self.device)
        return dict_data

    def get_vec(self, list_text, max_text_length, batch_size, num_workers):
        dataset = TextDataset(self.plm_path, list_text, max_text_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        data_iter = tqdm.tqdm(enumerate(dataloader), desc='get_vec', total=len(dataloader), bar_format="{l_bar}{r_bar}")
        list_vec = []
        for idx_batch, data in data_iter:
            data = self.dict_data2gpu(data)
            with torch.no_grad():
                h_cls = self.model(data['input_ids'], data['padding_mask'])[0][:, 0]
            batch_h_cls = h_cls.cpu().tolist()
            list_vec += batch_h_cls
        return list_vec



## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

class CorpusBenchmarkHeuristic(object):
    def __init__(self, seed=0, device='cuda:0', max_text_length=500, language='en'):
        assert language in ['en', 'zh'] # TODO:
        self.language = language
        self.seed = seed
        self.device = device
        random.seed(self.seed)

        self.max_text_length = max_text_length
        self.corpus2list_text = {}
        self.corpus2list_vec = {}

        self.list_entity = []
        self.keyword_processor = None

    def dedup_list_text(self, list_text, th=0.8):
        return list_text

    def sample_list_text(self, corpus_folder_path, text_key, sample_line_count=10000):
        # print('sample list text', corpus_folder_path, sample_line_count)
        list_dict_data = sample_data_without_cache(corpus_folder_path, int(1.5 * sample_line_count))

        list_text = [dict_data[text_key] for dict_data in list_dict_data]

        list_text_dedup = self.dedup_list_text(list_text)
        random.shuffle(list_text_dedup)

        list_text = list_text_dedup[:sample_line_count]
        # print('sample_list_text, len(list_text)', len(list_text))

        self.corpus2list_text[corpus_folder_path] = list_text
        return list_text

    # Heuristic Functions ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
    def length_distribution(self, corpus_folder_path, text_key, sample_line_count):
        print('length_distribution, corpus_folder_path', corpus_folder_path)

        if corpus_folder_path not in self.corpus2list_text:
            list_text = self.sample_list_text(corpus_folder_path, text_key, sample_line_count)
        else:
            list_text = self.corpus2list_text[corpus_folder_path]

        list_doc_length = [len(text.split()) for text in list_text]

        list_para_count = [len(text.replace('\n\n', '\n').split('\n')) for text in list_text]

        return list_doc_length, list_para_count
    def lexical_diversity(self, corpus_folder_path, text_key, sample_line_count):
        if corpus_folder_path not in self.corpus2list_text:
            list_text = self.sample_list_text(corpus_folder_path, text_key, sample_line_count)
        else:
            list_text = self.corpus2list_text[corpus_folder_path]
        if self.language == 'zh':
            list_text_cur = [' '.join(text[:self.max_text_length]) for text in list_text]
        else:
            list_text_cur = [' '.join(text.split()[:self.max_text_length]) for text in list_text]

        list_MTLD_score = []
        for str_words in list_text_cur:
            score = LexicalRichness(str_words).mtld(threshold=0.72) #  TODO:threshold
            list_MTLD_score.append(float(score))
        return list_MTLD_score


    def task2vec_diversity(self, corpus_folder_path, text_key, gpt2_name, batch_sample_size=512,
                           batch_sample_count=100, max_text_length=512, device='cuda:0'):
        list_batch_sample = sample_task2vec_corpus(corpus_folder_path, text_key, max_text_length,
                                                   batch_sample_size, batch_sample_count)

        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
        tokenizer.pad_token = tokenizer.eos_token  # '<|endoftext|>'
        tokenizer.pad_token_id = tokenizer.eos_token_id  # '<|endoftext|>'

        model = GPT2LMHeadModel.from_pretrained(gpt2_name).to(device)

        for p in model.parameters(): p.requires_grad = False
        for p in model.lm_head.parameters(): p.requires_grad = True

        list_embedding = []
        for sample_idx in range(batch_sample_count):
            dataset = DatasetTask2Vec(list_batch_sample[sample_idx], tokenizer)
            data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False)

            montecarlo_fisher(model, data_loader, epochs=1, device=device)
            embedding = extract_embedding(model)

            list_embedding.append(embedding)
        distance_matrix = pdist(list_embedding, distance='cosine')
        div, ci = stats_of_distance_matrix(distance_matrix)
        return div

    def knowledge_diversity_zh(self, corpus_folder_path, text_key,
                               sample_line_count, wiki_data_entity_path, plm_path):
        if corpus_folder_path not in self.corpus2list_text:
            list_text = self.sample_list_text(corpus_folder_path, text_key, sample_line_count)
        else:
            list_text = self.corpus2list_text[corpus_folder_path]

        total_char = sum([len(text) for text in list_text])

        if not self.keyword_processor:
            self.keyword_processor = KeywordProcessor()
            list_entity = []
            with open(wiki_data_entity_path, 'r') as f_read:
                for idx, line in tqdm.tqdm(enumerate(f_read), ncols=50):
                    dict_data = json.loads(line)
                    entity = dict_data['name_zh']
                    if len(entity) > 1:
                        list_entity.append(' ' + entity + ' ')
            self.keyword_processor.add_keywords_from_list(list_entity)
        else:
            pass

        dict_entity2count = {}
        for text in list_text:
            text_split = ' ' + ' '.join(jieba.lcut(text)) + ' '

            keywords_found = self.keyword_processor.extract_keywords(text_split)
            keywords_found = set(keywords_found)

            for entity in keywords_found:
                if entity not in dict_entity2count:
                    dict_entity2count[entity] = 0
                dict_entity2count[entity] += 1

        #### sort
        sorted_entity2count = sorted(dict_entity2count.items(), key=lambda x: x[1], reverse=True)


        np_count_entity = np.array([tuple_e2c[1] for tuple_e2c in sorted_entity2count])

        list_entity_occur = [tuple_e2c[0][1:-1] for tuple_e2c in sorted_entity2count]  # 去掉起止空格
        list_vec = SentenceVec(plm_path=plm_path). \
            get_vec(list_text=list_entity_occur, max_text_length=64, batch_size=64, num_workers=8)

        matrix_sim = cosine_similarity(list_vec)
        list_sim = matrix_sim[np.triu_indices_from(matrix_sim, k=1)]

        random.shuffle(list_sim)
        list_sim = list_sim[:sample_line_count * sample_line_count // 10]

        return list_sim, np.sum(np_count_entity) / (total_char + 1e-9)

    def knowledge_diversity_en(self, corpus_folder_path, text_key,
                               sample_line_count, wiki_data_entity_path, plm_path):
        if corpus_folder_path not in self.corpus2list_text:
            list_text = self.sample_list_text(corpus_folder_path, text_key, sample_line_count)
        else:
            list_text = self.corpus2list_text[corpus_folder_path]

        total_word = sum([len(text.split()) for text in list_text])

        if not self.keyword_processor:
            self.keyword_processor = KeywordProcessor()
            list_entity = []
            with open(wiki_data_entity_path, 'r') as f_read:
                for idx, line in tqdm.tqdm(enumerate(f_read), ncols=50):
                    dict_data = json.loads(line)
                    entity = dict_data['name_en']
                    if len(entity) > 1:
                        list_entity.append(' ' + entity + ' ')
            self.keyword_processor.add_keywords_from_list(list_entity)
        else:
            pass

        dict_entity2count = {}
        for text in list_text:
            text_split = ' ' + text + ' '

            keywords_found = self.keyword_processor.extract_keywords(text_split)
            keywords_found = set(keywords_found)

            for entity in keywords_found:
                if entity not in dict_entity2count:
                    dict_entity2count[entity] = 0
                dict_entity2count[entity] += 1

        #### sort
        sorted_entity2count = sorted(dict_entity2count.items(), key=lambda x: x[1], reverse=True)

        print('len(sorted_entity2count)', len(sorted_entity2count))

        np_count_entity = np.array([tuple_e2c[1] for tuple_e2c in sorted_entity2count])
        list_entity_occur = [tuple_e2c[0][1:-1] for tuple_e2c in sorted_entity2count]  # 去掉起止空格
        list_vec = SentenceVec(plm_path=plm_path). \
            get_vec(list_text=list_entity_occur, max_text_length=64, batch_size=64, num_workers=8)

        matrix_sim = cosine_similarity(list_vec)
        list_sim = matrix_sim[np.triu_indices_from(matrix_sim, k=1)]

        random.shuffle(list_sim)
        list_sim = list_sim[:sample_line_count * sample_line_count // 10]

        return list_sim, np.sum(np_count_entity) / (total_word + 1e-9)


    def wiki_distribution_similarity_zh(self, corpus_folder_path, text_key, sample_line_count, kenlm_path):
        if corpus_folder_path not in self.corpus2list_text:
            list_text = self.sample_list_text(corpus_folder_path, text_key, sample_line_count)
        else:
            list_text = self.corpus2list_text[corpus_folder_path]

        model_kenlm = kenlm.Model(kenlm_path)

        list_ppl = []
        for text in tqdm.tqdm(list_text, ncols=50, total=len(list_text)):
            text_lm = ' '.join(text.replace('\n', ' ').split())
            text_lm = ' '.join(list(text_lm))
            ppl = model_kenlm.perplexity(text_lm)
            list_ppl.append(ppl)

        return list_ppl


    def semantic_diversity(self, corpus_folder_path, text_key, sample_line_count, plm_path):
        if corpus_folder_path not in self.corpus2list_text:
            list_text = self.sample_list_text(corpus_folder_path, text_key, sample_line_count)
        else:
            list_text = self.corpus2list_text[corpus_folder_path]

        if corpus_folder_path not in self.corpus2list_vec:
            list_vec = SentenceVec(plm_path=plm_path). \
                get_vec(list_text=list_text, max_text_length=self.max_text_length, batch_size=16, num_workers=8)
        else:
            list_vec = self.corpus2list_vec[corpus_folder_path]

        matrix_sim = cosine_similarity(list_vec)
        list_sim = matrix_sim[np.triu_indices_from(matrix_sim, k=1)]

        random.shuffle(list_sim)
        list_sim = list_sim[:sample_line_count // 10]

        return list_sim

    def cluster_diversity(self, corpus_folder_path, text_key, sample_line_count, cluster_count, plm_path):
        if corpus_folder_path not in self.corpus2list_text:
            list_text = self.sample_list_text(corpus_folder_path, text_key, sample_line_count)
        else:
            list_text = self.corpus2list_text[corpus_folder_path]

        if corpus_folder_path not in self.corpus2list_vec:
            list_vec = SentenceVec(plm_path=plm_path). \
                get_vec(list_text=list_text, max_text_length=self.max_text_length, batch_size=16, num_workers=8)
        else:
            list_vec = self.corpus2list_vec[corpus_folder_path]

        k_means = KMeans(n_clusters=cluster_count, max_iter=400, init='k-means++', random_state=0)
        k_means.fit(list_vec)
        k_means_res = k_means.predict(list_vec)

        list_cluster_size = []
        list_cluster_center_vec = []
        for cluster_idx, vec_center in enumerate(k_means.cluster_centers_):
            list_data_idx_cluster = [data_idx_ for data_idx_, cluster_idx_ in enumerate(k_means_res)
                                     if cluster_idx_ == cluster_idx]

            list_dis = [np.linalg.norm(vec_center - list_vec[data_idx]) for data_idx in list_data_idx_cluster]
            idx_data_center = list_data_idx_cluster[list_dis.index(min(list_dis))]

            list_cluster_center_vec.append(list_vec[idx_data_center])
            list_cluster_size.append(len(list_data_idx_cluster))

        matrix_sim = cosine_similarity(list_cluster_center_vec)
        list_sim = matrix_sim[np.triu_indices_from(matrix_sim, k=1)]

        return list_sim, list_cluster_size

    # Heuristic Functions  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~  ~
    def get_meta_info(self, corpus_folder_path, text_key, sample_line_count, meta_info_file_name,
                      list_heuristic_function,
                      gpt2_name,
                      bert_name,
                      wiki_data_entity_path,
                      wiki_kenlm_path
                      ):
        print('get_meta_info', corpus_folder_path, meta_info_file_name, sample_line_count)

        list_text = self.sample_list_text(corpus_folder_path, text_key, sample_line_count)

        #### 0⃣️ length_distribution
        print('#### 0⃣️ length_distribution')
        list_doc_length, list_para_count = self.length_distribution(corpus_folder_path, text_key, sample_line_count)
        print('#### 0⃣️ length_distribution', np.percentile(np.array(list_doc_length), [0, 25, 50, 75, 100]))
        print('#### 0⃣️ length_distribution', np.percentile(np.array(list_para_count), [0, 25, 50, 75, 100]))

        #### 1⃣️ lexical_diversity
        print('#### 1⃣️ lexical_diversity')
        list_mtld_score = self.lexical_diversity(corpus_folder_path, text_key, sample_line_count)
        print('#### 1⃣️ lexical_diversity', np.percentile(np.array(list_mtld_score), [0, 25, 50, 75, 100]))

        #### 3⃣️ knowledge_diversity
        print('#### 3⃣️ knowledge_diversity')
        list_sim_knowledge, knowledge_density = self.knowledge_diversity_en(corpus_folder_path, text_key,
                               sample_line_count, wiki_data_entity_path, bert_name)
        print('#### 3⃣️ knowledge_diversity', knowledge_density)

        #### 4⃣️ wiki_distribution_similarity
        print('#### 4⃣️ wiki_distribution_similarity')
        list_kenlm_ppl = self.wiki_distribution_similarity_zh(corpus_folder_path, text_key, sample_line_count, wiki_kenlm_path)

        #### 5⃣️ semantic_diversity
        print('#### 5⃣️ semantic_diversity')
        list_sim_semantic = self.semantic_diversity(corpus_folder_path, text_key, sample_line_count, bert_name)

        #### 6⃣️ cluster_diversity
        print('#### 6⃣️ cluster_diversity')
        list_sim_cluster, list_cluster_size = self.cluster_diversity(corpus_folder_path, text_key, sample_line_count, 100, bert_name)


        #### 2⃣️ task2vec_diversity
        print('#### 2⃣️ task2vec_diversity')
        task2vec_div = self.task2vec_diversity(corpus_folder_path, text_key, gpt2_name,
            batch_sample_size=512, batch_sample_count=50, max_text_length=self.max_text_length, device=self.device) # TODO
        print('#### 2⃣️ task2vec_diversity', task2vec_div)

        file_path_write = 'scripts/assessment/info/{:s}'.format(meta_info_file_name)

        with open(file_path_write, 'w') as f_write:
            dict_info = {
                'list_doc_length': list(list_doc_length), 'list_para_count': list(list_para_count),

                'list_mtld_score': list(list_mtld_score),

                'task2vec_div': float(task2vec_div),

                'list_sim_knowledge': list(list_sim_knowledge), 'knowledge_density': float(knowledge_density),

                'list_kenlm_ppl': list(list_kenlm_ppl),

                'list_sim_semantic': list(list_sim_semantic),

                'list_sim_cluster': list(list_sim_cluster), 'list_cluster_size': list(list_cluster_size),

            }
            f_write.write(json.dumps(dict_info, ensure_ascii=False) + '\n')

        print('get_meta_info DONE!', corpus_folder_path, file_path_write)





if __name__ == '__main__':
    print('heuristic evaluation')
    benchmark = CorpusBenchmarkHeuristic(
        seed=0,
        device='cuda:0',
        max_text_length=512,
        language='en',
    )
    benchmark.get_meta_info(
        corpus_folder_path='/data/tongzhou/corpus/cc_rule-en/may-jun-2023.subset.high_quality/',
        text_key='content',
        sample_line_count=10000,
        meta_info_file_name='cc_en.may-jun-2023.wiki_vs_cc',
        list_heuristic_function=[],
        gpt2_name='gpt2',
        bert_name='bert-base-uncased',
        wiki_data_entity_path='/data/tongzhou/corpus/wikidata/wiki-entity.clean.idf-remain.txt',
        wiki_kenlm_path='/data/tongzhou/Web/streamlit_demo_v4/scripts/klm/wiki.en.word.5gram.klm',
    ) # high_quality
