from keras import backend as K, initializers, regularizers, constraints
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.initializers import TruncatedNormal
from keras.models import Model, load_model
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras.layers import *

def build_SRL_model(LEN_SEQUENCE, pre_word_embeddings, words, pos_tags, lemmas, roles):

    word_emb = Input(shape = (LEN_SEQUENCE,), dtype = "int32")
    pos_emb = Input(shape = (LEN_SEQUENCE,), dtype = "int32")
    lemma_emb = Input(shape = (LEN_SEQUENCE,), dtype = "int32")
    flag = Input(shape = (LEN_SEQUENCE,), dtype = "float32")
    role_em = Input(shape = (len(roles),), dtype = 'int32')

##############################################################################################################################################

    w = Embedding(len(words), 100, input_length = LEN_SEQUENCE, trainable = True, embeddings_initializer = TruncatedNormal())(word_emb)
    w2v = Embedding(len(words), 300, input_length = LEN_SEQUENCE, weights = [pre_word_embeddings], trainable = False)(word_emb)
    pos = Embedding(len(pos_tags), 16, input_length = LEN_SEQUENCE, trainable = True, embeddings_initializer = TruncatedNormal())(pos_emb)
    lemma = Embedding(len(lemmas), 100, input_length = LEN_SEQUENCE, trainable = True, embeddings_initializer = TruncatedNormal(), name = 'predicate')(lemma_emb)

##############################################################################################################################################

    new_flag = Lambda(lambda x:K.expand_dims(x, axis=-1))(flag)

    input_0 = Concatenate(axis=-1)([w, w2v, pos, lemma, new_flag])

##############################################################################################################################################

    hid = 512

    hs = Bidirectional(CuDNNLSTM(hid, return_sequences = True, kernel_initializer = TruncatedNormal()))(input_0)
    hs = Bidirectional(CuDNNLSTM(hid, return_sequences = True, kernel_initializer = TruncatedNormal()))(hs)
    hs = Bidirectional(CuDNNLSTM(hid, return_sequences = True, kernel_initializer = TruncatedNormal()))(hs)
    hs = Bidirectional(CuDNNLSTM(hid, return_sequences = True, kernel_initializer = TruncatedNormal()))(hs)

##############################################################################################################################################

    predicate_h = multiply([hs, new_flag])

    predicate_h_ = Lambda(lambda x:K.sum(x, axis=1))(predicate_h)

    predicate_ = RepeatVector(LEN_SEQUENCE)(predicate_h_)

    new_hs = Concatenate(axis=-1)([predicate_, hs])

##############################################################################################################################################

    lemma_out = Embedding(len(lemmas), 128, input_length = LEN_SEQUENCE, trainable = True, embeddings_initializer = TruncatedNormal())(lemma_emb)

    predicate_em = multiply([lemma_out, new_flag])

    predicate_em_ = Lambda(lambda x:K.sum(x, axis=1))(predicate_em)

    predicate_e = RepeatVector(len(roles))(predicate_em_)

    role_emb = Embedding(len(roles), 128, input_length = len(roles), trainable = True, embeddings_initializer = TruncatedNormal())(role_em)

    input_1 = Concatenate(axis=-1)([predicate_e, role_emb])

    w = TimeDistributed(Dense(hid*4, activation ='relu', kernel_initializer = TruncatedNormal()))(input_1)

##############################################################################################################################################

    out = Lambda(lambda x:K.batch_dot(x[0], x[1], axes=[2,2]))([new_hs, w])

    pred = Activation('softmax')(out)

##############################################################################################################################################

    model = Model(inputs = [word_emb, pos_emb, lemma_emb, flag, role_em], outputs = pred)

    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.001))

    return model