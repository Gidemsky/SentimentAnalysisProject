from PyDictionary import PyDictionary
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
import gensim.downloader as api


# creates df of words and weights (like pearson cor df)
def get_word_weights(fname):
    df = pd.read_csv(fname)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df


def short_words_synonyms(df):
    dictionary = PyDictionary()
    # finding synonyms of short words first
    have_short_words = df.loc[df['short_word'].notna()]
    # dropping column in order to drop duplicates afterwards and not repeat multiple calls to synonym for same word
    have_short_words.drop(columns=['word'], inplace=True)
    have_short_words.drop_duplicates(inplace=True)
    have_short_words.reset_index(drop=True, inplace=True)
    # comment if short words results are saved
    have_short_words['synonyms'] = have_short_words['short_word'].apply(lambda x: dictionary.synonym(x))
    have_short_words.drop(columns=['p_cor'], inplace=True)
    return have_short_words


# creates 'short word' of words with endings - only for english
def shorten_word(df):
    # finding similar word
    df['to_change'] = df.word.str.endswith(('ily', 'ly', 'ive', 'iest', 'ier', 'able', 'ably', 'able2', 'ably2'))
    df['short_word'] = df['word']
    to_append = df.loc[df['word'].str.endswith(('ably', 'able'))]
    to_append['word'] = to_append['word'] + '2'
    # in order to add prove from provable or provably as well as adapt from adaptable or adaptably
    df = df.append(to_append)
    df['short_word'] = np.where(df['word'].str.endswith('ly'), df['word'].str[:-2], df['short_word'])
    df['short_word'] = np.where(df['word'].str.endswith('ive'), df['word'].str[:-3] + 'e', df['short_word'])
    df['short_word'] = np.where(df['word'].str.endswith('iest'), df['word'].str[:-4] + 'y', df['short_word'])
    df['short_word'] = np.where(df['word'].str.endswith('ier'), df['word'].str[:-3] + 'y', df['short_word'])
    # had to do these ones after 'ly' so it wouldn't mess up words like hastily, amiably
    df['short_word'] = np.where(df['word'].str.endswith('ily'), df['word'].str[:-3] + 'y', df['short_word'])
    df['short_word'] = np.where(df['word'].str.endswith(('ably', 'able')), df['word'].str[:-4], df['short_word'])
    df['short_word'] = np.where(df['word'].str.endswith(('ably2', 'able2')), df['word'].str[:-5] + 'e',
                                df['short_word'])
    df['word'] = np.where(df['word'].str.endswith('2'), df['word'].str[:-1], df['word'])

    df['short_word'] = np.where(df['to_change'] == False, np.nan, df['short_word'])
    df.drop(columns=['to_change'], inplace=True)
    return df


def short_words_weight_to_full_vocab(full_df, short_df):
    merged = pd.merge(short_df, full_df, how='left', left_on='short_word', right_on='word')
    # dropping word_y because its the same as short_word
    merged.drop(columns=['p_cor_x', 'word_y'], inplace=True)
    merged.rename(columns={'word_x': 'word', 'p_cor_y': 'p_cor'}, inplace=True)
    full_df = pd.merge(full_df, merged, how='left', on='word')
    full_df.rename(columns={'p_cor_x': 'p_cor'}, inplace=True)
    # replacing zeros in p_cor with new value from short word p_cor
    full_df['p_cor'] = np.where(full_df['p_cor_y'] > 0, full_df['p_cor_y'], full_df['p_cor'])
    full_df.drop(columns=['p_cor_y', 'short_word'], inplace=True)
    full_df.sort_values(by=['word', 'p_cor'], inplace=True)
    full_df.drop_duplicates(subset='word', keep='last', inplace=True)
    return full_df


# finds short version of long word in vocab (surprisingly - surprise)
def get_other_synonyms(full_df, short_words_df):
    # want same synonyms for identical words (that were found on short words but happen to be a full word as well)
    full_df_merged = pd.merge(full_df, short_words_df, how='left', left_on='word', right_on='short_word')
    # want same synonyms for words with similar meaning
    # example: word= happily, short_word_x = happy, short_word_y = happy
    full_df_merged.drop(columns=['short_word_y'], inplace=True)
    full_df_merged.rename(columns={'short_word_x': 'short_word'}, inplace=True)
    full_df_merged = pd.merge(full_df_merged, short_words_df, how='left', left_on='short_word', right_on='short_word')
    full_df_merged['synonyms_x'] = np.where(full_df_merged.synonyms_y.notna() & full_df_merged.synonyms_x.isna(),
                                            full_df_merged.synonyms_y, full_df_merged.synonyms_x)
    full_df_merged.drop(columns='synonyms_y', inplace=True)
    full_df_merged.rename(columns={'synonyms_x': 'synonyms'}, inplace=True)
    have_syns = full_df_merged.loc[full_df_merged['synonyms'].notna()]
    no_syns = full_df_merged.loc[full_df_merged['synonyms'].isna()]
    dictionary = PyDictionary()
    no_syns['synonyms'] = no_syns['word'].apply(lambda x: dictionary.synonym(x))
    have_syns = have_syns.append(no_syns)
    return have_syns


# finds weight average from similar words
def avg_synonym_weight(ls, have_weights_df):
    if str(ls) == 'nan':
        return 0
    try:
        ls = ls.split(', ')
    except:
        c = 1
    sum = 0
    num = 0
    p_cor = have_weights_df.loc[have_weights_df['word'].isin(ls)]['p_cor']
    if p_cor.empty:
        return 0
    return p_cor.mean()


# creates new pos words by word letter similarity (only for english)
def expand_vocab_w_syns(fname):
    vocab_words = get_word_weights(fname)
    zero_p_cor = vocab_words.loc[vocab_words['p_cor'] == 0]
    zero_p_cor = shorten_word(zero_p_cor)
    # giving weight of short words to similar words
    vocab_words = short_words_weight_to_full_vocab(vocab_words, zero_p_cor)
    zero_p_cor = vocab_words.loc[vocab_words['p_cor'] == 0]
    zero_p_cor = shorten_word(zero_p_cor)
    # in comment because i already saved restults of short_words_synonyms
    z_short_words = short_words_synonyms(zero_p_cor)
    # z_short_words = pd.read_csv('vocab_classifier/results/aug_11_train/pos_short_syns.csv')
    # in comment because i already saved results of full_z_df_syns
    full_z_df_syns = get_other_synonyms(zero_p_cor, z_short_words)
    # full_z_df_syns = pd.read_csv('vocab_classifier/results/aug_11_train/neg_all_train_syns.csv')
    # fixing string representation of list in order to convert to list later
    full_z_df_syns.replace({'\'': '', '\[': '', '\]': ''}, regex=True, inplace=True)
    have_weight = vocab_words.loc[vocab_words['p_cor'] > 0]
    # converting string to list
    full_z_df_syns['p_cor'] = full_z_df_syns['synonyms'].apply(
        lambda x: avg_synonym_weight(x, have_weight))
    vocab_words = have_weight
    vocab_words = vocab_words.append(full_z_df_syns)
    # sorting by word and weight so last of each word will be with highest weight (0 or higher than 0)
    vocab_words.sort_values(by=['word', 'p_cor'], inplace=True)
    vocab_words.drop_duplicates(subset='word', keep='last', inplace=True)
    vocab_words.reset_index(drop=True, inplace=True)
    return vocab_words


# w2v dif
def cal_vec_dif(word, other_words, word_vectors):
    sim_word = ''
    similarity = 0
    sim = 0
    for w in other_words:
        if w == word:
            continue
        try:
            sim = word_vectors.similarity(word, w)
            # saving word with max similarity to current word
            if sim > similarity:
                sim_word = w
                similarity = sim
        except:
            continue
    return sim_word, similarity


# get p_cor of similar word
def get_p_cor_from_sim(word, sim_word_vals_df):
    if str(word) == 'nan':
        return 0
    new_val = sim_word_vals_df.loc[sim_word_vals_df['word'] == word]['p_cor']
    if new_val.empty:
        return 0
    return float(new_val)


# finds similar words within words list and takes best match weight
def calc_word_to_vec(words_df):
    words_df['word_2'] = words_df['word']
    # in comment if already saved df in previous run
    # only want similarities to words with weight greater than 0

    word_vectors = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
    words_df['similarity'] = words_df['word'].apply(
        lambda x: cal_vec_dif(x, words_df.loc[words_df['p_cor'] > 0]['word_2'], word_vectors))
    words_df[['word_sim', 'similarity']] = pd.DataFrame(words_df['similarity'].tolist(), index=words_df.index)
    words_df.drop(columns=['word_2'], inplace=True)

    # words_df = pd.read_csv('vocab_classifier/results/real_data/trans_pos_after_word_vec_func_11_09.csv')
    words_df.sort_values(by=['word', 'p_cor'], ascending=False, inplace=True)
    # should only be relevant for translated vocabulary because multiple words are translated to the same word
    words_df.drop_duplicates(subset='word', keep='first', inplace=True)
    # need to save words df for some reason as csv and then read and the can run next line
    words_df['new_p_cor'] = words_df['word_sim'].apply(lambda x: get_p_cor_from_sim(x, words_df[['word', 'p_cor', 'similarity']]))
    words_df['p_cor'] = np.where(words_df['p_cor'] == 0, words_df['new_p_cor'], words_df['p_cor'])
    words_df.drop(columns=['new_p_cor'], inplace=True)
    # save df
    return words_df


# extends weights by finding word2vec similarity
def add_w2v_p_cors(pos_fname, neg_fname):
    pos_words_df = get_word_weights(pos_fname)
    pos_words_df = calc_word_to_vec(pos_words_df)
    neg_words_df = get_word_weights(neg_fname)
    neg_words_df = calc_word_to_vec(neg_words_df)
    # save weights!
    return pos_words_df, neg_words_df


if __name__ == '__main__':
    new_pos_words, new_neg_words = add_w2v_p_cors(
        'vocab_classifier/results/real_data/trans_pos_pear_11_9.csv',
        'vocab_classifier/results/real_data/trans_neg_pear_11_9.csv')

    a = 1
