from evaluation import Scorer
from conll import *
from srl_model import *
from util import *

def train(LEN_SEQUENCE, file_train, corpus, labels_role, words, pos_tags, lemmas, pre_word_embeddings, freq, alpha, batch_size, epochs):
    
    X_train, y_train, active_pred, true_pos, true_lemma, sequence_lengths, role_emb = get_input_files(corpus, words, labels_role, LEN_SEQUENCE, pos_tags, lemmas, freq, alpha, True)

    print('Training..')

    model = build_SRL_model(LEN_SEQUENCE, pre_word_embeddings, words, pos_tags, lemmas, labels_role)

    filepath='../SRLData/srl/model-{epoch:02d}-.h5'
    
    checkpoint = ModelCheckpoint(filepath)

    tbCallBack = TensorBoard(log_dir='./logs', histogram_freq = 1, write_graph = True, update_freq = 'batch')

    callbacks_list = [checkpoint, tbCallBack]

    model.fit([X_train, true_pos, true_lemma, active_pred, role_emb], y_train, validation_split = 0.1, batch_size = batch_size, epochs = epochs, callbacks = callbacks_list)

    return model

def test(LEN_SEQUENCE, file_test, model, words, labels_role, pos_tags, lemmas):

    print('Testing..')

    corpus = extract_information(file_test, LEN_SEQUENCE, False)

    X_test, y_test, active_pred, true_pos, true_lemma, sequence_lengths, role_emb = get_input_files(corpus, words, labels_role, LEN_SEQUENCE, pos_tags, lemmas, None, None, False)

    y_test = np.argmax(y_test , axis=-1)

    y_pred = np.argmax(model.predict([X_test, true_pos, true_lemma, active_pred, role_emb]), axis=-1)

    new_test = []
    new_pred = []

    for x, y, z in zip(y_test, y_pred, sequence_lengths):
        new_test.append(x[:z])
        new_pred.append(y[:z])

    evaluate = Scorer()

    evaluate(new_test, new_pred)

    precision, recall, f1_score = evaluate.compute_metrics()

    return precision, recall, f1_score

def run():

    file_train = '../SRLData/EN/CoNLL2009-ST-English-train.txt'

    file_test = '../SRLData/EN/CoNLL2009_en_gold.txt'

    LEN_SEQUENCE = 70

    batch_size = 32

    epochs = 5

    alpha = 0.25

    print('Start !')

    corpus, labels_role, words, pos_tags, lemmas, freq = extract_information(file_train, LEN_SEQUENCE, True)

    #pre_word_embeddings = get_emb_matrix(words)

    with open('../SRLData/embeddings/'+str(LEN_SEQUENCE)+'w2v.pickle', 'rb') as handle:
        pre_word_embeddings = pickle.load(handle)

    print('Get Train files..')

    #model = train(LEN_SEQUENCE, file_train, corpus, labels_role, words, pos_tags, lemmas, pre_word_embeddings, freq, alpha, batch_size, epochs)

    model = load_model('../SRLData/srl/'+str(LEN_SEQUENCE)+'.h5')

    precision, recall, f1_score = test(LEN_SEQUENCE, file_test, model, words, labels_role, pos_tags, lemmas)

    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1_score))


if __name__ == '__main__':
    run()
