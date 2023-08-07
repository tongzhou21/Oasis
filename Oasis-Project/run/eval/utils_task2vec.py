import torch.nn.functional as F
import tqdm
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import json
import os

def get_subfiles_abs_path(corpus_folder_path):
    list_abs_path = []
    for root, dirs, files in os.walk(corpus_folder_path, topdown=False):
        for name in files:
            abs_path = os.path.join(root, name)
            list_abs_path.append(abs_path)
    return list_abs_path

class DatasetTask2Vec(Dataset):
    def __init__(self, list_text, tokenizer):
        self.tokenizer = tokenizer
        self.list_text = list_text
        self.max_seq_len = 512
        print('DatasetTask2Vec len(list_text)', len(list_text))

    def __len__(self):
        return len(self.list_text)

    def __getitem__(self, item):
        str_text = self.list_text[item]

        dict_inp = self.tokenizer(str_text, max_length=self.max_seq_len, truncation=True)

        dict_inp['input_ids'] = dict_inp['input_ids'][:self.max_seq_len]
        dict_inp['attention_mask'] = dict_inp['attention_mask'][:self.max_seq_len]

        dict_inp['input_ids'] += [self.tokenizer.pad_token_id] * (self.max_seq_len - len(dict_inp['input_ids']))
        dict_inp['attention_mask'] += [0] * (self.max_seq_len - len(dict_inp['attention_mask']))

        dict_inp['input_ids'] = torch.tensor(dict_inp['input_ids'])
        dict_inp['attention_mask'] = torch.tensor(dict_inp['attention_mask'])
        dict_inp['labels'] = dict_inp['input_ids'] # TODO: pad IGNORE_INDEX

        return dict_inp


def dict_data2gpu(dict_data, device):
    for key, value in dict_data.items():
        if isinstance(value, dict):
            dict_data[key] = dict_data2gpu(value)
        else:
            dict_data[key] = value.to(device)
    return dict_data

def montecarlo_fisher(model, data_loader, epochs=1, device='cuda:0'):
    for p in model.parameters():
        p.grad2_acc = torch.zeros_like(p.data)
        p.grad_counter = 0

    for k in range(epochs):
        for i, batch_data in tqdm.tqdm(enumerate(data_loader), desc="Computing Fisher"):
            batch_data = dict_data2gpu(batch_data, device)
            output = model(**batch_data)
            logits = output.logits

            target = torch.multinomial(F.softmax(logits.reshape(-1, logits.size(-1)), dim=-1), 1).detach().view(-1)

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target)

            model.zero_grad()

            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad2_acc += p.grad.data ** 2
                    p.grad_counter += 1
    for p in model.parameters():
        if p.grad_counter == 0:
            del p.grad2_acc
        else:
            p.grad2_acc /= p.grad_counter
    print('montecarlo_fisher done')

class Embedding:
    def __init__(self, hessian, scale, meta=None):
        self.hessian = np.array(hessian)
        self.scale = np.array(scale)
        self.meta = meta

def extract_embedding(model):
    hess, scale = [], []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'grad2_acc'):
            print('extract_embedding module', module)
            grad2 = module.weight.grad2_acc.cpu().detach().numpy()
            filterwise_hess = grad2.reshape(grad2.shape[0], -1).mean(axis=1)
            hess.append(filterwise_hess)
            scale.append(np.ones_like(filterwise_hess))
    return Embedding(hessian=np.concatenate(hess), scale=np.concatenate(scale), meta=None)


import scipy.spatial.distance as distance

_DISTANCES = {}

def _register_distance(distance_fn):
    _DISTANCES[distance_fn.__name__] = distance_fn
    return distance_fn
def get_hessian(e, normalized=False):
    hess = np.array(e.hessian)
    if normalized:
        scale = np.array(e.scale)
        hess = hess / scale
    return hess

def get_hessians(*embeddings, normalized=False):
    return [get_hessian(e, normalized=normalized) for e in embeddings]

def get_scaled_hessian(e0, e1):
    h0, h1 = get_hessians(e0, e1, normalized=False)
    return h0 / (h0 + h1 + 1e-8), h1 / (h0 + h1 + 1e-8)
@_register_distance
def cosine(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    return distance.cosine(h1, h2)
def pdist(embeddings, distance='cosine'):
    distance_fn = _DISTANCES[distance]
    n = len(embeddings)
    distance_matrix = np.zeros([n, n])
    if distance != 'asymmetric_kl':
        for (i, e1), (j, e2) in itertools.combinations(enumerate(embeddings), 2):
            distance_matrix[i, j] = distance_fn(e1, e2)
            distance_matrix[j, i] = distance_matrix[i, j]
    else:
        for (i, e1) in enumerate(embeddings):
            for (j, e2) in enumerate(embeddings):
                distance_matrix[i, j] = distance_fn(e1, e2)
    return distance_matrix

def get_diagonal(matrix: np.ndarray,
                 check_if_symmetric: bool = False,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triu: np.ndarray = np.triu(matrix)
    tril: np.ndarray = np.tril(matrix)
    # distance_matrix = distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)].reshape(distance_matrix.shape[0], -1)
    # remove diagonal and dummy zeros where the other triangular matrix was artificially placed.
    distance_matrix: np.ndarray = triu[triu != 0.0]

    # - check we are returning diagonal so sit's samller than full matrix
    size_full_matrix: int = matrix.size
    size_diag: int = distance_matrix.size
    assert size_diag < size_full_matrix, f'The diagonal matrix is not being extracted correct:' \
                                         f'\n{size_diag=}, {size_full_matrix=}'

    # - flatten
    flatten: np.ndarray = distance_matrix.flatten()
    # - check is of size (N,)
    assert flatten.shape == (flatten.shape[0],)
    assert len(flatten.shape) == 1
    assert isinstance(flatten.shape[0], int)
    return flatten, triu, tril


def mean_confidence_interval(data: iter, confidence: float = 0.95) -> tuple[float, np.ndarray]:
    import scipy.stats
    import numpy as np
    # - move tensor to cpu and numpy if not already there
    if isinstance(data, torch.Tensor):
        # chatgpt says last bit might not be needed but seems popular so left it
        data = data.detach()
        data: np.ndarray = data.cpu().numpy() if data.is_cuda else data.numpy()

    a: np.ndarray = 1.0 * np.array(data)
    n: int = len(a)
    if n == 1:
        import logging
        logging.warning('The first dimension of your data is 1, perhaps you meant to transpose your data? or remove the'
                        'singleton dimension?')
    m, se = a.mean(), scipy.stats.sem(a)
    tp = scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    h = se * tp
    return m, h

def approx_equal(val1: float, val2: float, tolerance: float = 1.0e-4) -> bool:
    eq: bool = abs(val1 - val2) <= tolerance
    return eq

def stats_of_distance_matrix(distance_matrix: np.ndarray,
                             remove_diagonal: bool = True,
                             variance_type: str = 'ci_0.95',
                             ddof: int = 1,
                             ) -> tuple[float, float]:
    if remove_diagonal:
        flatten, triu, tril = get_diagonal(distance_matrix)
    else:
        flatten: np.ndarray = distance_matrix.flatten()

    # - compute stats of distance matrix
    if variance_type == 'std':
        mu, var = flatten.mean(), flatten.std(ddof=ddof)
    elif variance_type == 'ci_0.95':
        mu, var = mean_confidence_interval(flatten, confidence=0.95)
    else:
        raise ValueError(f'Invalid variance type, got: {variance_type=}')

    if remove_diagonal:
        assert approx_equal(triu.sum(), tril.sum(),
                            tolerance=1e-4), f'Distance matrix is not symmetric, are you sure this is correct?'
        assert approx_equal(flatten.mean(), triu[triu != 0.0].mean(),
                            tolerance=1e-4), f'Mean should be equal to triangular matrix'
        assert approx_equal(mu, triu[triu != 0.0].mean(), tolerance=1e-4)
    return mu, var


def sample_task2vec_corpus(corpus_folder_path, text_key, max_text_len, batch_sample_size, batch_sample_count):
    list_batch_sample = [[] for _ in range(batch_sample_count)]
    flag_add = True

    def add_text2bach_sample_cat(list_batch_sample, batch_sample_size, text, max_text_len=512):
        list_batch_size = [len(batch_samples) for batch_samples in list_batch_sample]
        list_cand_batch_idx = [idx for idx, batch_size in enumerate(list_batch_size) if
                               batch_size < batch_sample_size]
        flag_add = True
        if len(list_cand_batch_idx) == 0:
            flag_add = False

        if flag_add:
            batch_idx = random.choice(list_cand_batch_idx)
            list_batch_sample[batch_idx].append(text[:max_text_len])
        else:  # 都满了 查看长度不足的序列
            for batch_idx in range(len(list_batch_sample)):
                for sample_idx in range(len(list_batch_sample[batch_idx])):
                    if len(list_batch_sample[batch_idx][sample_idx]) < max_text_len:
                        list_batch_sample[batch_idx][sample_idx] += '\n\n' + text
                        list_batch_sample[batch_idx][sample_idx] = list_batch_sample[batch_idx][sample_idx][
                                                                   :max_text_len]
                        flag_add = True
                        break
                if flag_add: break

        return list_batch_sample, flag_add

    def sample_pretrain_corpus(list_batch_sample, text_key, max_text_len):
        list_file_path = get_subfiles_abs_path(corpus_folder_path)
        random.shuffle(list_file_path)
        flag_add = True
        for file_path in tqdm.tqdm(list_file_path, total=len(list_file_path), ncols=64):
            with open(file_path, 'r') as f_read:
                for line in tqdm.tqdm(f_read, ncols=32):  # TODO: random skip
                    text = json.loads(line)[text_key]
                    list_batch_sample, flag_add = add_text2bach_sample_cat(list_batch_sample, batch_sample_size, text, max_text_len)

                    if flag_add == False: break
                if flag_add == False: break
        return list_batch_sample, flag_add

    list_batch_sample, flag_add = sample_pretrain_corpus(list_batch_sample, text_key=text_key, max_text_len=max_text_len)
    return list_batch_sample

