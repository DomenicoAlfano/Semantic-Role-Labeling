from collections import defaultdict
import itertools
import sys
import re

def read_sentences(data):
    for key, group in itertools.groupby(data, lambda x: not x.strip()):
        if not key:
            yield list(group)

def process(frame_ids, arguments, frames):
    arguments = list(zip(*arguments))
    assert (len(arguments) == len(frame_ids))

    frame_data = defaultdict(list)
    for frame_id, arg, targ in zip(frame_ids, arguments, frames):
        roles = [label if label != '_' else 'O' for label in arg]
        frm = [l if l != '_' else 'O' for l in targ]

        for i, x in enumerate(frm):
            if x != 'O':
                idx = i

        frame_data['f_' + frame_id].append({
            'roles': roles,
            'frames': frm,
            'target': {
                'index': [list(range(idx, idx + 1))]
            }
        })

    return frame_data

def from_conll(block, dict_words, dict_labels_pos, dict_labels_role, dict_lemmas, dict_words_freq):
    record = defaultdict(list)
    block_frames, block_arguments, block_targets = [], [], []
    for line in block:
        parts = re.split("\t", line.strip())
        word = parts[1].lower()
        pos_tag = parts[5]
        if parts[13] != '_':
            predicate = parts[13].lower()
            block_targets.append(['_' for i in block])
            block_targets[-1][int(parts[0]) - 1] = predicate
        else:
            predicate = '_'
        record['tokenized_sentence'].append(word)
        record['pos'].append(pos_tag)

        if predicate not in dict_lemmas and predicate != '_':
            dict_lemmas[predicate] = len(dict_lemmas)

        if word not in dict_words:
            dict_words[word] = len(dict_words)

        if word not in dict_words_freq:
            dict_words_freq[word] = 1
        else:
            dict_words_freq[word] += 1

        if pos_tag not in dict_labels_pos:
            dict_labels_pos[pos_tag] = len(dict_labels_pos)

        arguments = parts[14:]
        for a in arguments:
            if a not in dict_labels_role and a != '_':
                dict_labels_role[a] = len(dict_labels_role)

        if predicate != '_':
            block_frames.append(predicate)

        block_arguments.append(arguments)

    frame_data = process(block_frames, block_arguments, block_targets)
    record.update(frame_data)
    return record, dict_labels_role, dict_words, dict_labels_pos, dict_lemmas, dict_words_freq

def extract_information(file, max_input_dim, label):

    corpus = defaultdict(dict)

    dict_words = dict()
    dict_labels_pos = dict()
    dict_lemmas = dict()
    dict_labels_role = dict()
    dict_words_freq = dict()

    dict_labels_pos['PAD'] = len(dict_labels_pos)
    dict_lemmas['PAD'] = len(dict_lemmas)
    dict_lemmas['O'] = len(dict_lemmas)
    dict_lemmas['UNK'] = len(dict_lemmas)
    dict_labels_role['PAD'] = len(dict_labels_role)
    dict_labels_role['O'] = len(dict_labels_role)
    dict_words['PAD'] = len(dict_words)
    dict_words['UNK'] = len(dict_words)

    data = open(file)
    for doc_id, block in enumerate(read_sentences(data)):
        if len(block) <= max_input_dim:
            corpus[doc_id][doc_id], labels_role, words, labels_pos, lemmas, freq = from_conll(block, dict_words, dict_labels_pos, dict_labels_role, dict_lemmas, dict_words_freq)
        else:
            continue
    
    if label == True:
        return corpus, labels_role, words, labels_pos, lemmas, freq
    else:
        return corpus
