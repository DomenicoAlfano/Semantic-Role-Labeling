from keras.preprocessing.sequence import pad_sequences
import gensim.models.keyedvectors as word2vec
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import itertools
import string
import pickle

def to_emb_vector(model, word, num):
    
    try:
        return model[word]
    except KeyError:
        return np.random.normal(0.0, 0.1, num)

def get_emb_matrix(words):

    print('Embeddings..')
    
    w2v = word2vec.KeyedVectors.load_word2vec_format('../SRLData/emb_model/GoogleNews-vectors-negative300.bin', binary = True)

    pre_word_emb = dict()
    
    for word in words.keys():
        pre_word_emb[word] = to_emb_vector(w2v, word, 300)

    pre_word_embeddings = np.asarray(list(pre_word_emb.values()))
    print(pre_word_embeddings.shape)

    # with open('../SRLData/embeddings/70w2v.pickle', 'wb') as handle:
    #     pickle.dump(pre_word_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print('Saved')
    
    return pre_word_embeddings

def frame_data(data):
    data = OrderedDict(sorted(data.items()))
    for doc in data:
        for sentence in data[doc]:
            for field in data[doc][sentence]:
                if field.startswith('f_'):
                    for frame_instance in data[doc][sentence][field]:
                        yield doc, sentence, field[2:], frame_instance

def get_input_files(data, words, labels_role, max_input_dim, pos_tags, lemmas, freq, alpha, flag):

    X = []
    Y = []

    true_predicate = []
    true_pos = []
    true_lemmas = []
    sequence_lengths = []
    indices = []
    role_emb = []

    if flag:
        new_words = dict()

        for word in words.keys():
            if word == 'UNK':
                new_words[word] = words['UNK']
            elif word == 'PAD':
                new_words[word] = words['PAD']
            elif word in lemmas.keys():
                new_words[word] = words[word]
            else:
                r = np.random.random()
                prob = alpha / (freq[word] + alpha)
                if r > prob:
                    new_words[word] = words[word]
                else:
                    new_words[word] = words['UNK']

    for doc_id, sent_id, frame_name, frame_instance in frame_data(data):
        x = []
        y = []
        pos = []
        lemma = []
        sen = []

        for fr in data[doc_id][sent_id]['tokenized_sentence']:
            if flag == True:
                x.append(new_words[fr])
            else:
                if fr in words.keys():
                    x.append(words[fr])
                else:
                    x.append(words['UNK'])
        
        for p in data[doc_id][sent_id]['pos']:
            pos.append(pos_tags[p])

        true_pred = np.array([1.0 if frame_instance['target']['index'][0][0] == i else 0.0 for i in range(len(x))])
        
        for role in frame_instance['roles']:
            y.append(labels_role[role])

        for pred in frame_instance['frames']:
            if pred in lemmas.keys():
                lemma.append(lemmas[pred])
            else:
                lemma.append(lemmas['UNK'])

        X.append(x)
        Y.append(y)
        true_predicate.append(true_pred)
        true_pos.append(pos)
        true_lemmas.append(lemma)
        sequence_lengths.append(len(data[doc_id][sent_id]['tokenized_sentence']))
        role_emb.append(list(range(len(labels_role))))

    X_t = pad_sequences(X, maxlen = max_input_dim, dtype='int64', padding='post')
    Y_t = to_categorical(pad_sequences(Y, maxlen = max_input_dim, dtype='int64', padding='post'), num_classes=len(labels_role))
    true_pos = pad_sequences(true_pos, maxlen = max_input_dim, dtype='int64', padding='post')
    true_lemmas = pad_sequences(true_lemmas, maxlen = max_input_dim, dtype='int64', padding='post')
    true_predicate = pad_sequences(true_predicate, maxlen = max_input_dim, dtype='int64', padding='post')

    return np.array(X_t), np.array(Y_t), np.array(true_predicate), np.array(true_pos), np.array(true_lemmas), np.array(sequence_lengths), np.array(role_emb)