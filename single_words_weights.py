from PyDictionary import PyDictionary
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np


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


def expand_vocab_w_syns(fname):
    vocab_words = get_word_weights(fname)
    zero_p_cor = vocab_words.loc[vocab_words['p_cor'] == 0]
    zero_p_cor = shorten_word(zero_p_cor)
    # giving weight of short words to similar words
    vocab_words = short_words_weight_to_full_vocab(vocab_words, zero_p_cor)
    zero_p_cor = vocab_words.loc[vocab_words['p_cor'] == 0]
    zero_p_cor = shorten_word(zero_p_cor)
    # in comment because i already saved restults of short_words_synonyms
    # z_short_words = short_words_synonyms(zero_p_cor)
    # z_short_words = pd.read_csv('vocab_classifier/results/neg_only_short_words_synonyms.csv')
    # in comment because i already saved results of full_z_df_syns
    # full_z_df_syns = get_other_synonyms(zero_p_cor, z_short_words)
    full_z_df_syns = pd.read_csv('vocab_classifier/results/neg_full_df_synonyms.csv')
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


if __name__ == '__main__':
    # pos_words = expand_vocab_w_syns('vocab_classifier/results/pos_words_pearson_train.csv')
    neg_words = expand_vocab_w_syns('vocab_classifier/results/neg_words_pearson_train.csv')

    a = 1
