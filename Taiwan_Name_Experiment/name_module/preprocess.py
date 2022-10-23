from gensim.models import word2vec
from hanziconv import HanziConv
from name_module.share_lib import is_number, is_chinese, reduce_mem_usage
from name_module.Chinese_name_separate import *
from name_module.fortune_map_calculate import *
import pickle
import pandas as pd
import traceback
from pathlib import Path as plib_path


def drop_dirty_name(name_df, fb_name_df=pd.DataFrame()):
    id_col = 'userID'
    if id_col in fb_name_df.columns:
        # Drop name from duplicate user from FB source
        fb_name_df = name_df[name_df[id_col].apply(
            lambda x: not isinstance(x, float))]
        ori_len = len(fb_name_df)
        fb_name_df = fb_name_df.drop_duplicates(
            subset=id_col, keep='first', inplace=True)
        print('Drop duplicate FB name: from ', ori_len, '->',
              len(fb_name_df), ' drop:', ori_len - len(fb_name_df))

        name_df = name_df[name_df[id_col].apply(
            lambda x: isinstance(x, float))]
        name_df = pd.concat([name_df, fb_name_df], ignore_index=True)

    # Drop Message is not number
    ori_len = len(name_df)
    name_df = name_df[name_df.message.apply(
        lambda x: not isinstance(x, float))]
    name_df = name_df[name_df.message.apply(lambda x: is_number(x))]
    print('Drop Message is not number: ', ori_len, '->',
          len(name_df), ' drop:', ori_len - len(name_df))

    # Drop English name
    ori_len = len(name_df)
    name_df = name_df[name_df.name.apply(lambda x: (is_chinese(x)))]
    print('Drop English name: from ', ori_len, '->',
          len(name_df), ' drop:', ori_len - len(name_df))

    ori_len = len(name_df)
    # Drop last name is not in Taiwan all last name
    name_df = name_df[name_df.name.apply(lambda x: (check_last_name(x)))]
    print('Drop last name is not in Taiwan last name list :', ori_len, '->', len(name_df), ' drop:',
          ori_len - len(name_df))
    ori_len = len(name_df)

    name_df['LastName'] = name_df.name.apply(lambda x: (get_last_name(x)))
    name_df['FirstName'] = name_df.name.apply(lambda x: (get_first_name(x)))

    # Drop First name is longer than 3
    name_df = name_df[name_df.FirstName.apply(lambda x: len(x) < 3)]
    print('Drop First name is longer than 3  :', ori_len,
          '->', len(name_df), ' drop:', ori_len - len(name_df))

    return name_df


def load_saved_files():
    """ Load saved files for add feature.
    Returns:
        all_character_in_name(list): all character in name
        moe_data_dict(dict): dict of moe data
        synonyms_dict(dict): dict of synonyms words
        consonants(list): list of consonants in Hanyu_Pinyin
        vowels(list): list of vowels in Hanyu_Pinyin
        special_word_dict(dict): dict of special word
    """
    pkl_data_path = './pkl_data/'
    '''
    w2v phonetic
    '''
    # 2XX萬名字的涵蓋的中文字 List,共8578種字, 如果把character 變成 編號的時候可以用
    with open(pkl_data_path + 'all_character_in_name_list.pkl', 'rb') as handle:
        all_character_in_name = pickle.loads(handle.read())

    # 開源萌典的本體 dict
    with open(pkl_data_path + 'moe_data_dict.pkl', 'rb') as handle:
        moe_data_dict = pickle.loads(handle.read())

    # 存同義字或是詞的dict
    with open(pkl_data_path + 'synonyms_dict.pkl', 'rb') as handle:
        synonyms_dict = pickle.loads(handle.read())

    '''
    phonetic
    '''
    with open(pkl_data_path + 'Hanyu_Pinyin_Consonant_list.pkl', 'rb') as handle:
        consonants = pickle.loads(handle.read())

    with open(pkl_data_path + 'Hanyu_Pinyin_Vowel_list.pkl', 'rb') as handle:
        vowels = pickle.loads(handle.read())

    with open(pkl_data_path + 'special_word_dict.pkl', 'rb') as handle:
        special_word_dict = pickle.loads(handle.read())

    return all_character_in_name, moe_data_dict, synonyms_dict, consonants, vowels, special_word_dict


def get_x_feature(name_feature, columns):
    """ Get selected category name features

    Args:
        name_feature (selected name): name feature
        columns (index): df columns

    Returns:
        feature_columns (list): selected columns
    """
    feature_columns = []
    if 'gender' in name_feature:
        feature_columns.extend(['Male_prob', 'Female_prob'])

    if 'uni-gram' in name_feature:
        feature_columns.extend(['FN1', 'FN2'])

    for col in columns:
        if 'W2V' in name_feature and '_wv_' in col:
            feature_columns.append(col)
        elif 'Phonetic' in name_feature and ('_Vowel' in col or '_Consonant' in col):
            feature_columns.append(col)
        elif 'Fortune_map' in name_feature and ("格_" in col or '三才_' in col):
            feature_columns.append(col)
        elif 'Zodiac' in name_feature and 'Zodiac_' in col:
            feature_columns.append(col)
        elif 'Radical' in name_feature and 'Radical_' in col:
            feature_columns.append(col)
    return feature_columns


def turn_all_word_in_w2v_model_to_dataframe(moe_model):
    """ Turn all word in w2v model to dataframe.

    Args:
        moe_model (gensim.models.word2vec.Word2Vec): word2vec model

    Returns:
        w2v_df (df): w2v dataframe
    """
    w2v_df = pd.DataFrame()
    for word in moe_model.wv.vocab:
        w2v_df[word] = moe_model.wv[word]
    return w2v_df.T


def add_word_vector(vector_model, word, wv_index, synonyms, vector_mean):
    """ Get word or synonyms vector value at location wv_index, if not exist, return vector_mean.
    Args:
        vector_model (genism.models.word2vec.Word2Vec): word2vec model
        word (str): character or word
        wv_index (int): index of vector
        synonyms (dict): dict of synonyms words
        vector_mean (list): vector mean

    Returns:
        vector_value (float): vector value
    """
    if word in vector_model.wv:
        return vector_model.wv[word][wv_index]
    else:
        if word in synonyms:
            # use same meaning word as substitute
            if synonyms[word] in vector_model.wv:
                return vector_model.wv[synonyms[word]][wv_index]
            else:
                return vector_mean[wv_index]
        else:
            return vector_mean[wv_index]


def get_first_name_word_vector(vector_model, first_name, index, wv_index, synonyms, w2v_mean):
    """ Get word vector for first name

    Args:
        vector_model (gensim.models.word2vec.Word2Vec): word2vec model
        first_name (str): first name
        index (int): index of first name
        wv_index (int): index of word vector
        synonyms (dict): synonyms dict
        w2v_mean (list): vector mean

    Returns:
        first_name_vector (float): cell in word vector
    """
    character = " "
    if len(first_name) == 2:
        character = first_name[index]
    elif index == 1:
        character = first_name
    return add_word_vector(vector_model, character, wv_index, synonyms, w2v_mean)


def add_w2v_feature(name_df, w2v_vector_number, synonyms, w2v_mean):
    """ Use pretrained word2vec to add word embedding feature.

    Args:
        name_df (df): Name dataframe
        w2v_vector_number (int): w2v vector number (vector length)
        synonyms (dict): synonyms dict
        w2v_mean (list): vector mean

    Returns:
        name_df (df): Name dataframe with w2v feature
        w2v_feature_columns (list): w2v feature columns
    """

    '''
     pretrained word2vec model, used wiki doc and moe dictionary word2vec
     every entry is a Chinese character, not a word
    '''
    moe_model = word2vec.Word2Vec.load("./w2v_data/wiki_moe_100_model.bin")

    for wv_index in range(0, w2v_vector_number):
        name_df['FN1_wv_{}'.format(wv_index)] = name_df["FirstName"].apply(
            lambda x:  get_first_name_word_vector(moe_model, x, 0, wv_index, synonyms, w2v_mean))
        name_df['FN2_wv_{}'.format(wv_index)] = name_df["FirstName"].apply(
            lambda x:  get_first_name_word_vector(moe_model, x, 1, wv_index, synonyms, w2v_mean))

    name_df, na_list = reduce_mem_usage(name_df)

    w2v_feature = get_x_feature(['W2V'], name_df.columns)
    print("w2v_feature len", len(w2v_feature))
    return name_df, w2v_feature


def seperate_pinyin(pinyin, vowels):
    """ Seperate pinyin to vowel and consonant.

    Args:
        pinyin (str): pinyin
        vowels (list): pinyin vowels list

    Returns:
        vowel (str): vowel
        consonant (str): consonant
    """

    if '（' in pinyin:
        pinyin = pinyin.replace("（讀音）", "").replace('（語音）', "")
        if " （" in pinyin:
            print(pinyin, "- with unkown pattern")
    for vowel in vowels:
        if vowel in pinyin:
            consonant = pinyin[:pinyin.index(vowel)]
            return vowel, consonant


def get_vowel_consonant(term, vowels, moe_data_dict, special_word_dict):
    """ Get vowel and consonant for term.
    Args:
        term (str): term in dictionary
        vowels (list): pinyin vowels list
        moe_data_dict(dict): moe data dict, main dictionary
        special_word_dict (dict): complementary word dictionary

    Returns:
        vowel (str): vowel
        consonant (str): consonant
    """
    vowel, consonant = "", ""
    if term == " ":
        return vowel, consonant

    try:
        if term not in moe_data_dict:
            # Convert simplified chinese character to traditional.
            term = HanziConv.toTraditional(term)

        if term in moe_data_dict:
            for hete in moe_data_dict[term]['heteronyms']:
                pinyin = hete.get('pinyin', None)
                if pinyin is not None:
                    vowel, consonant = seperate_pinyin(pinyin, vowels)
                    return vowel, consonant

                # 找不到字音，看是否是哪個字的異體字
                for define in hete['definitions']:
                    if '異體字' in define['def']:
                        d = define['def']
                        alter_term = d[d.index('「') + 1: d.index('」')]
                        # print(alt_term,term)

                        for alter_hete in moe_data_dict[alter_term]['heteronyms']:
                            pinyin = alter_hete.get('pinyin', None)
                            if pinyin is not None:
                                vowel, consonant = seperate_pinyin(
                                    pinyin, vowels)
                                return vowel, consonant
            # print('在字典內但沒有拼音：',term)
        else:
            # print('不在字典內：',term)
            if term in special_word_dict:
                pinyin = special_word_dict[term].get('pinyin', None)
                if pinyin is not None:
                    vowel, consonant = seperate_pinyin(pinyin, vowels)
                    return vowel, consonant
            else:
                print('拼音不明：', term)
        return vowel, consonant
    except Exception as e:
        print(traceback.format_exc())


def add_phonetic_feature(name_df, vowels, moe_data_dict, special_word_pinyin_dic):
    """ Add phonetic feature to dataframe.

    Args:
        name_df (pd.DataFrame): name dataframe
        vowels (list): pinyin vowels list
        moe_data_dict(dict): moe data dict, main dictionary
        special_word_pinyin_dic (dict): complementary word dictionary

    Returns:
        name_df (pd.DataFrame): name dataframe with phonetic feature
        phonetic_feature_columns (list): phonetic feature columns
    """
    one_hot_columns = []
    for i in range(1, 3, 1):
        character_consonant = []
        character_vowel = []
        for name in name_df["FirstName"].values:
            character = name[i - 1]
            vowel, consonant = get_vowel_consonant(
                character, vowels, moe_data_dict, special_word_pinyin_dic)
            character_consonant.append(consonant)
            character_vowel.append(vowel)
        name_df["FN{}_Consonant".format(i)] = character_consonant
        name_df["FN{}_Vowel".format(i)] = character_vowel
        one_hot_columns.append("FN{}_Consonant".format(i))
        one_hot_columns.append("FN{}_Vowel".format(i))

    name_df = pd.get_dummies(name_df, columns=one_hot_columns)
    phonetic_feature = get_x_feature(['Phonetic'], name_df.columns)
    print('phonetic_feature len:', len(phonetic_feature))

    return name_df, phonetic_feature


def add_fortune_map_feature(name_df, moe_data_dict, special_word_dict):
    """ Add fortune map feature to dataframe.

    Args:
        name_df (pd.DataFrame): name dataframe
        moe_data_dict (dict): main_dictionary
        special_word_dict (dict): additional_dictionary

    Returns:
        name_df (pd.DataFrame): name dataframe
        fortune_map_feature_columns (list): fortune map feature columns
    """
    fc = FortuneMapCalculater(moe_data_dict, special_word_dict)
    first_names = name_df["FirstName"].tolist()
    last_names = name_df["LastName"].tolist()
    name_df['天格'] = [fc.get_state_heaven(x) for x in last_names]
    name_df['地格'] = [fc.get_state_earth(x) for x in last_names]
    mans = []
    outsides = []
    totals = []
    talents = []
    for i in range(len(first_names)):
        mans.append(fc.get_state_man(last_names[i], first_names[i]))
        outsides.append(fc.get_state_outside(last_names[i], first_names[i]))
        totals.append(fc.get_state_total(last_names[i], first_names[i]))
        talents.append(fc.get_state_talent(last_names[i], first_names[i]))
    name_df['人格'] = mans
    name_df['外格'] = outsides
    name_df['總格'] = totals
    name_df['三才'] = talents

    name_df = pd.get_dummies(
        name_df, columns=["天格", "地格", "人格", "外格", "總格", "三才"])

    fortune_map_feature_list = get_x_feature(['Fortune_map'], name_df.columns)
    print("len on fortune_map_feature_list:", len(fortune_map_feature_list))
    return name_df, fortune_map_feature_list


def get_radical_column(character, dictionary):
    """ Add radical column to dataframe.

    Args:
        character (_type_): _description_
        dictionary (_type_): _description_

    Returns:
        _type_: _description_
    """
    if character in dictionary:
        return dictionary[character]['radical']
    else:
        return '不明'


def add_radical_feature(name_df, moe_data_dict):
    """ Add radical column to dataframe.
    """
    name_df["FN1_Radical"] = name_df['FirstName'].apply(
        lambda x: get_radical_column(x[0], moe_data_dict))
    name_df["FN2_Radical"] = name_df['FirstName'].apply(
        lambda x: get_radical_column(x[1], moe_data_dict))
    name_df = pd.get_dummies(name_df, columns=["FN1_Radical", "FN2_Radical"])
    radical_feature = get_x_feature(['Radical'], name_df.columns)
    print("len of Radical_feature_list: ", len(radical_feature))
    return name_df, radical_feature


def get_zodiac_from_birthyear(birthyear):
    zodiacs = ('鼠', '牛', '虎', '兔', '龍', '蛇', '馬', '羊', '猴', '雞', '狗', '豬')
    return zodiacs[divmod((birthyear - 4), 12)[1]]


def add_zodiac_feature(name_df):
    name_df['Zodiac'] = name_df['message'].apply(
        lambda x: get_zodiac_from_birthyear(x))
    name_df = pd.get_dummies(name_df, columns=["Zodiac"])

    zodiac_feature_list = get_x_feature("Zodiac", name_df.columns)
    print("len of Zodiac_feature_list: ", len(zodiac_feature_list))
    return name_df, zodiac_feature_list


def add_uni_gram(character, all_character_in_name):
    if character in all_character_in_name:
        return all_character_in_name.index(character)
    else:
        return -1


def preprocess(name_df, save_path=None, file_name=None):
    name_df = drop_dirty_name(name_df)

    """
    Add name feature
    """
    all_character_in_name, moe_data_dict, synonyms_dict, consonants, vowels, special_word_dict = load_saved_files()

    try:
        # W2V
        print("Add W2V feature")
        moe_model = word2vec.Word2Vec.load("./w2v_data/wiki_moe_100_model.bin")
        vector_mean = turn_all_word_in_w2v_model_to_dataframe(moe_model).mean()
        name_df, w2v_feature = add_w2v_feature(
            name_df, w2v_vector_number=100, synonyms=synonyms_dict, w2v_mean=vector_mean)

        # Phonetic - one-hot encoding
        print("Add phonetic feature")
        name_df, phonetic_feature = add_phonetic_feature(
            name_df, vowels, moe_data_dict, special_word_dict)

        # Fortune map - one-hot encoding
        print("Add fortune map feature")
        name_df, fortune_map_feature = add_fortune_map_feature(
            name_df, moe_data_dict, special_word_dict)

        # Character
        name_df['FN1'] = name_df.FirstName.apply(
            lambda x: add_uni_gram(x[0], all_character_in_name))
        name_df['FN2'] = name_df.FirstName.apply(
            lambda x: add_uni_gram(x[1], all_character_in_name))

        # Radical - one-hot encoding
        print("Add radical feature")
        name_df, radical_feature = add_radical_feature(name_df, moe_data_dict)

        # Zodiac - one-hot encoding
        print("Add zodiac feature")
        name_df, zodiac_feature = add_zodiac_feature(name_df)
        name_df, na_list = reduce_mem_usage(name_df)
    except:
        print(traceback.format_exc())

    if save_path is not None:
        name_df.to_csv(save_path / file_name.replace(".csv",
                       "_featured.csv"), index=False)
    return name_df


def rename_old_name_df_dict(name_df):
    rename_dic = {}
    for col in name_df.columns:
        if "radical" in col:
            rename_dic[col] = col.replace("radical", "Radical")
        elif "sonin" in col:
            if "sonin_-1" in col:
                rename_dic[col] = col.replace("sonin_-1", "Consonant_")
            else:
                rename_dic[col] = col.replace("sonin", "Consonant")
        elif "mu_in" in col:
            rename_dic[col] = col.replace("mu_in", "Vowel")
    name_df = name_df.rename(columns=rename_dic)
    return name_df
