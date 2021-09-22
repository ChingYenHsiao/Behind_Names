# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import csv
import os
import sys
import pickle

#from NameModules.fortune_calculator import moe_data_dict, special_word_dict
from NameModules.fortune_calculator import stroke_total
from NameModules.fortune_calculator import stroke_outside
from NameModules.fortune_calculator import stroke_man
from NameModules.fortune_calculator import stroke_heaven
from NameModules.fortune_calculator import stroke_earth
from NameModules.fortune_calculator import test_name_Fortune_telling
from NameModules.fortune_calculator import get_talent_type
from NameModules.fortune_calculator import get_talent_state
from NameModules.fortune_calculator import get_stroke_state
from NameModules.fortune_calculator import Get_stroke
from NameModules.share_lib import is_chinese, is_number, PrintException
from NameModules.Taiwan_name_seperate import checkLastName, GetLastName, GetFirstName, get_LastName_from_FN, is_biFirstName


'''
checkLastName: 檢查名字的姓氏有無在百家姓的list中
GetLastName: 取姓氏
GetLastName: 取名字
get_LastName_from_FN:
is_biFirstName:是否是疊字名字 
'''
from NameModules.Taiwan_name_seperate import Taiwan_LastName_List, Taiwan_LastName_len1_List, Taiwan_LastName_len2_List
from NameModules.NameModule import contain_Simplified_character , contain_unreadable_character

'''
contain_Simplified_character: find名字是否含非繁體字會用之字的簡體字
contain_unreadable_character: 名字否含不可解讀的編碼的特殊字元
'''

from NameModules.share_lib import is_number,is_chinese, reduce_mem_usage
'''
reduce_mem_usage: 自動將df各欄的型態轉型成可接受的較小size 型態
'''

from gensim.models import word2vec

from hanziconv import HanziConv 


def NamePreprecoess(Name_df):
    ID_col = 'userID'
    if ID_col in Name_df.columns:
        #Drop duplicate name in FB source
        FB_Name_df = Name_df [ Name_df[ID_col].apply(lambda x: type(x)!=float) ]
        ori_len = len(FB_Name_df)
        FB_Name_df = FB_Name_df.drop_duplicates(subset= ID_col ,keep='first',inplace=False)
        print ('Drop duplicate FB name: from ',ori_len,'->',len(FB_Name_df),' drop:',ori_len-len(FB_Name_df))
        
        Name_df = Name_df[ Name_df[ID_col].apply(lambda x: type(x)==float) ]
        Name_df = pd.concat([Name_df, FB_Name_df], ignore_index=True)
    
    #Drop Message is not number
    ori_len = len(Name_df)
    Name_df  = Name_df[ Name_df.message.apply(lambda x:type(x)!=float) ]
    Name_df  = Name_df[ Name_df.message.apply(lambda x: is_number(x)) ]
    print ('Drop Message is not number: ',ori_len,'->',len(Name_df),' drop:',ori_len-len(Name_df))
    ori_len = len(Name_df)
    
    #Drop English name
    ori_len = len(Name_df)
    Name_df = Name_df[ Name_df.name.apply(lambda x: (is_chinese(x))) ]
    print ('Drop English name: from ',ori_len,'->',len(Name_df),' drop:',ori_len - len(Name_df))

    ori_len = len(Name_df)
    #Drop last name is not in Taiwan all last name
    Name_df = Name_df[Name_df.name.apply(lambda x: ( checkLastName(x)))]
    print ('Dron last name is not in Taiwan last name list :',ori_len,'->',len(Name_df),' drop:',ori_len - len(Name_df))
    ori_len = len(Name_df)

    Name_df['LastName'] = Name_df.name.apply(lambda x: (GetLastName(x)))
    Name_df['FirstName'] = Name_df.name.apply(lambda x: (GetFirstName(x)))

    #Drop First name is longer than 3 
    Name_df = Name_df[Name_df.FirstName.apply(lambda x: len(x)<3 )]
    print ('Drop First name is longer than 3  :',ori_len,'->',len(Name_df),' drop:',ori_len - len(Name_df))
    ori_len = len(Name_df)
    
    return Name_df


def Read_Saved_Files():
    pkl_data_path = './pkl_data/'
    '''
    w2v phonetic
    '''
    # 2XX萬名字的涵蓋的中文字 List,共8578種字, 如果把character 變成 編號的時候可以用
    Totalname_list = ''
    with open( pkl_data_path + 'Totalname_list.pkl', 'rb') as handle:
        Totalname_list = pickle.loads(handle.read())

    # 萌典的本體 dict
    moe_data_dict = {}
    with open( pkl_data_path +'moe_data_dict.pkl', 'rb') as handle:
        moe_data_dict = pickle.loads(handle.read())

    # 存同義字或是詞的dict
    common_dict = {}
    with open( pkl_data_path + 'common_dict.pkl', 'rb') as handle:
        common_dict = pickle.loads(handle.read())

    '''
    phonetic
    '''
    son_in_list = []
    with open( pkl_data_path + 'son_in_list.pkl', 'rb') as handle:
        son_in_list = pickle.loads(handle.read())

    mu_in_list = []
    with open( pkl_data_path + 'mu_in_list.pkl', 'rb') as handle:
        mu_in_list = pickle.loads(handle.read())

    special_word_dict = {}
    with open( pkl_data_path + 'special_word_dict.pkl', 'rb') as handle:
        special_word_dict = pickle.loads(handle.read())

    return Totalname_list, moe_data_dict, common_dict, son_in_list, mu_in_list, special_word_dict

def get_x_feature(feature_list , columns):
    f = []

    if 'gender' in feature_list:
        f+=['Male_prob' , 'Female_prob']
        
    for col in columns:
        if 'w2v' in feature_list and '_wv_' in col:
            f+= [col]
        elif 'phonetic' in feature_list and ( 'muin_' in col or 'sonin_' in col):
            f+=[col]
        elif 'fortune_map' in feature_list and ("格_" in col or '三才_' in col):
            f+=[col]
        elif 'Zodiac' in feature_list and 'Zodiac_' in col:
            f+=[col]
        elif 'radical' in feature_list and 'radical_' in col:
            f+=[col]
            
    return f 

def add_word_vector(vector_model,word,n):
    if word in vector_model.wv:
        return vector_model.wv[word][n]
    else:
        if word in common_dict:
            if common_dict[word] in vector_model.wv:
                return vector_model.wv[common_dict[word]][n]
            else:
                return 0
        else:
            return 0
        
def Get_FN_WV( vector_model, FirstName, FN_loc ,WV_loc):
    FN =' '
    if len(FirstName)==2:
        if FN_loc == 1:FN=FirstName[0]
        else:FN=FirstName[1]
    elif FN_loc==2:
        FN=FirstName
    return add_word_vector( vector_model,FN,WV_loc)       

def character_to_index(name,n):
    if n==1:
        if name[0] in Totalname_list:
            return Totalname_list.index(name[0])
        else:
            Totalname_list.append(name[0])
            return Totalname_list.index(name[0])
    if n==2 and len(name)==2:
        if name[1] in Totalname_list:
            return Totalname_list.index(name[1])
        else:
            Totalname_list.append(name[1])
            return Totalname_list.index(name[1])
    return -1

def add_pin_in_column(character, mode):
    # mode 1 = sonin = consonant
    # mode 2 = muin = vowel

    if character == -1:
        return None
    
    #不在字典內的補齊
    specail_word_pinyin_dic = {'艶': 'yàn', '鳯': 'fèng', '恵': 'huì', '姈': 'líng', '寳': 'bǎo', '姫': 'jī', '鑅': 'róng',
                               "玂": "qí", "浤": "hóng", '煊': 'xuān', '斔': 'zhōng', '琜': 'lái', '苰': 'hóng', '玹': 'xuán', '姵': 'pèi', '妏': 'wèn',
                               '妘': 'yún', '珺': 'jùn', '媗': 'xuān', '彣': 'wén', '玹': 'xuán', '瀞': 'jìng', '妡': 'xīn', '琁': 'xuán', '浤': 'hóng', '緁': 'jī',
                               '媜': 'zhēng', '姸': 'yán', '嬅': 'huà', '眞': 'zhēn', '廼': 'nǎi', '寛': 'kuān', '秝': 'lì', '蕥': 'yǎ', '汯': 'hóng', '逹': 'dá', '萓': 'yí',
                               '媃': 'róu', '孋': 'lí', '媁': 'wěi', '祤': 'yǔ', '媄': 'měi', '夆': 'fēng', '蒝': 'yuán', '嬣': 'níng', '砡': 'yù', '芠': 'wén',
                               '姳': 'mǐng', '蔆': 'líng', '菈': 'lā', '鍹': 'xuān', '榳': 'tíng', '錤': 'jī', '憓': 'huì', '潓': 'huì', '瓈': 'lí', '芛': 'wěi',
                               '峮': 'qún', '鋕': 'zhì', '姷': 'yòu', '兪': 'yú', '瑠': 'liú', '嫙': 'xuán', '珅': 'shēn', '暟': 'kǎi', '斈': 'xué', '煐': 'yīng', '淓': 'fāng', '瑨': 'jìn', '嬨': 'cí', '琹': 'qín', '珆': 'yí', '琣': 'pěi',
                               '娪': 'wú', '荺': 'yǔn', '爕': 'xiè', '玶': 'píng', '鋆': 'yún', '愼': 'shèn', '斳': 'qín', '瑈': 'róu', '澪': 'líng', '珦': 'xiàng', '妶': 'xián', '姃': 'zhēng', '薾': 'ěr', '溎': 'guì', '琄': 'xuàn', '琡': 'shū', '瑭': 'táng', '嫆': 'róng'
                               }
    #term = character
    term = Totalname_list[character]
    unkown_dict = {}
    try:
        if term not in moe_data_dict:
            term = HanziConv.toTraditional(term)

        if term == ' ':
            return None

        if term in moe_data_dict:
            for hete in moe_data_dict[term]['heteronyms']:
                if 'pinyin' in hete and moe_data_dict[term]['title'] != '啐':
                    word_p = hete['pinyin']

                    if '（' in (hete['pinyin']):
                        if '（讀音）' in hete['pinyin']:
                            word_p = hete['pinyin'].replace('（讀音）', '')
                        if '（語音）' in hete['pinyin']:
                            word_p = hete['pinyin'].replace('（語音）', '')
                        if '(' in word_p:
                            print(word_p+"!!")

                    for mu in mu_in_list:
                        if mu in word_p:
                            if mode == 'sonin':
                                return word_p[: word_p.index(mu)]
                            else:
                                return mu
            # 找不到字音，看是否是哪個字的異體字
            for hete in moe_data_dict[term]['heteronyms']:
                for define in hete['definitions']:
                    if '異體字' in define['def']:
                        d = define['def']
                        alt_term = d[d.index('「')+1: d.index('」')]
                        # print(alt_term,term)

                        for hete2 in moe_data_dict[alt_term]['heteronyms']:
                            if 'pinyin' in hete2 and moe_data_dict[alt_term]['title'] != '啐':

                                word_p = hete2['pinyin']
                                if '（' in (hete2['pinyin']):
                                    if '（讀音）' in hete2['pinyin']:
                                        word_p = hete2['pinyin'].replace(
                                            '（讀音）', '')
                                    if '（語音）' in hete2['pinyin']:
                                        word_p = hete2['pinyin'].replace(
                                            '（語音）', '')
                                    if '(' in word_p:
                                        print(word_p+"!!")

                                for mu in mu_in_list:
                                    if mu in word_p:
                                        if mode == 'sonin':
                                            return word_p[: word_p.index(mu)]
                                        else:
                                            return mu
            # print('在字典內但沒有拼音：',term)
        else:
            # print('不在字典內：',term)

            if term in special_word_dict:
                word_p = special_word_dict[term]['pinyin']
                for mu in mu_in_list:
                    if mu in word_p:
                        if mode == 'sonin':
                            return word_p[: word_p.index(mu)]
                        else:
                            return mu
            else:
                print('拼音不明：',term)
                if term not in unkown_dict:
                    unkown_dict[term] = 1
                else:
                    unkown_dict[term] += 1

#             if len( moe_df[moe_df.字詞名.apply(lambda x: x==term)])>0:
#                 print('不在字典內：',term)
        return unkown_dict
    except Exception as e:
        print(e)
        PrintException()

def add_W2V_feature( Name_df,w2v_Vector_number):
    ###############################################################
    #Add w2v
    #用17G的維基百科+幾十M的萌典範文訓練的w2v, 每個entry是一個字
    moe_model = word2vec.Word2Vec.load("./w2v_data/wiki_moe_100_model.bin")
    
    for i in range(0,w2v_Vector_number):
        Name_df['FN1_wv_'+str(i)]  = Name_df["FirstName"].apply(lambda x:  Get_FN_WV( moe_model , x, 1 , i))
        Name_df['FN2_wv_'+str(i)]  = Name_df["FirstName"].apply(lambda x:  Get_FN_WV( moe_model , x, 2 , i))
        
    Name_df , NA_list = reduce_mem_usage( Name_df )
    
    w2v_feature = get_x_feature( ['w2v'], Name_df.columns)
    print("w2v_feature len",len(w2v_feature))
    ################################################################ 
    return Name_df , w2v_feature

def add_phonetic_feature(Name_df):
    Name_df['FN1'] = Name_df.FirstName.apply(lambda x: character_to_index(x,1))
    Name_df['FN2'] = Name_df.FirstName.apply(lambda x: character_to_index(x,2))
    
    Name_df['FN1_sonin'] = Name_df.FN1.apply(lambda x:  add_pin_in_column(x,'sonin') )
    Name_df['FN1_muin'] = Name_df.FN1.apply(lambda x:  add_pin_in_column(x,'muin') )

    Name_df['FN2_sonin'] = Name_df.FN2.apply(lambda x:  add_pin_in_column(x,'sonin') )
    Name_df['FN2_muin'] = Name_df.FN2.apply(lambda x:  add_pin_in_column(x,'muin') )

    Name_df = pd.get_dummies(Name_df, columns=["FN1_sonin"])
    Name_df = pd.get_dummies(Name_df, columns=["FN1_muin"])
    Name_df = pd.get_dummies(Name_df, columns=["FN2_sonin"])
    Name_df = pd.get_dummies(Name_df, columns=["FN2_muin"])
    
    phonetic_feature = get_x_feature( ['phonetic'] , Name_df.columns)
    print('phonetic_feature len:',len(phonetic_feature))
    
    return Name_df , phonetic_feature

def add_fortune_map_feature(sampled_df):
    sampled_df['天格'] = sampled_df['LastName'].apply(lambda x: get_stroke_state(stroke_heaven (x)) )
    sampled_df['地格'] = sampled_df['LastName'].apply(lambda x: get_stroke_state(stroke_earth (x)) )
    sampled_df['人格'] = sampled_df.apply(lambda x: get_stroke_state(stroke_man (x)) ,axis = 1)
    sampled_df['外格'] = sampled_df.apply(lambda x: get_stroke_state(stroke_outside (x)) ,axis = 1)
    sampled_df['總格'] = sampled_df.apply(lambda x: get_stroke_state(stroke_total (x)) ,axis = 1)
    sampled_df['三才'] = sampled_df.apply(lambda x: get_talent_state(get_talent_type(stroke_heaven(
        x['LastName'])) + get_talent_type(stroke_earth(x['LastName'])) + get_talent_type(stroke_man(x))), axis=1)
    
    sampled_df = pd.get_dummies(sampled_df, columns=["天格"])
    sampled_df = pd.get_dummies(sampled_df, columns=["地格"])
    sampled_df = pd.get_dummies(sampled_df, columns=["人格"])
    sampled_df = pd.get_dummies(sampled_df, columns=["外格"])
    sampled_df = pd.get_dummies(sampled_df, columns=["總格"])
    sampled_df = pd.get_dummies(sampled_df, columns=["三才"])

    fortune_map_feature_list = get_x_feature( ['fortune_map'] , sampled_df.columns)
    print("len on fortune_map_feature_list:",len(fortune_map_feature_list))
    return sampled_df , fortune_map_feature_list

def add_radical_column_age(character):
    if character in moe_data_dict:
        return moe_data_dict[character]['radical'] 
    elif character in special_word_dict :
        return special_word_dict[character]['radical']  
    else:
        return '不明'
        #return -1

def add_radical_feature(sampled_df):
    #FN1存成index
    sampled_df['FN1_radical'] = sampled_df['FN1'].apply(lambda x: add_radical_column_age( Totalname_list[x]))
    sampled_df['FN2_radical'] = sampled_df['FN2'].apply(lambda x: add_radical_column_age( Totalname_list[x]))
    sampled_df = pd.get_dummies(sampled_df, columns=["FN1_radical"])
    sampled_df = pd.get_dummies(sampled_df, columns=["FN2_radical"])

    Radical_feature_list = get_x_feature( ['radical'] , sampled_df.columns)
    print("len of Radical_feature_list: ",len(Radical_feature_list))
    return sampled_df , Radical_feature_list
    
def get_zodiac_from_birthyear(birthyear):
    zodiac_list = ['鼠', '牛', '虎', '兔', '龍', '蛇', '馬', '羊', '猴', '雞', '狗', '豬']
    return zodiac_list[divmod((birthyear - 4), 12)[1]]


def add_zodiac_feature(Name_df):
    Name_df['Zodiac'] = Name_df['message'].apply(lambda x: get_zodiac_from_birthyear(x))
    Name_df = pd.get_dummies(Name_df, columns=["Zodiac"])
    
    Zodiac_feature_list = get_x_feature("Zodiac" , Name_df )
    print("len of Zodiac_feature_list: ",len(Zodiac_feature_list))
    return Name_df , Zodiac_feature_list 

if __name__ == "__main__":

    Path = './NameData/' 
    FileName = 'gcname_df'
    Name_df = pd.read_csv(Path + FileName + '.csv', dtype='str')

    Name_df = Name_df[[ 'name', 'BirthYear', 'FirstName', 'LastName', 'gender',
       'message', 'userID']]
    
    
    #將object type 復原回int
    Name_df['BirthYear'] = Name_df.BirthYear.apply(lambda x: int(x))
    Name_df['gender'] = Name_df.gender.apply(lambda x: int(float(x)))
    Name_df['message'] =  Name_df.message.apply(lambda x: int(x))

    #自動轉型
    Name_df , NA_list = reduce_mem_usage( Name_df)
    
    print("Name_df len",len(Name_df))
    Name_df = NamePreprecoess(Name_df)
    
    Totalname_list, moe_data_dict, common_dict, son_in_list, mu_in_list, special_word_dict = Read_Saved_Files()
 
    '''
    1 #用17G的維基百科+幾十M的萌典範文訓練的w2v, 每個entry是一個字
    '''
    
    Name_df , w2v_feature = add_W2V_feature( Name_df,w2v_Vector_number = 100)
    
    ################################################################
    '''
    2. 加入拼音feature  - one-hot encoding
    拼音是根據萌典與新華字典切出來的
    子音 = consonant = sonin 
    母音 = vowel = muin
    之前是隨意取名
    '''
    Name_df , phonetic_feature = add_phonetic_feature( Name_df)
    
    '''
    3. Fortune map - one-hot encoding
    '''
    Name_df , fortune_map_feature_list = add_fortune_map_feature ( Name_df )
    
    '''
    4. Character - one-hot encoding
    radical
    '''   
    Name_df , radical_feature_list = add_radical_feature ( Name_df )

    '''
    5. Zodiac_feature - one-hot encoding
    '''
    Name_df, Zodiac_feature_list = add_zodiac_feature( Name_df )
    
    Name_df.to_csv(Path + FileName + '_featured.csv', index= 0)
    

def Name_df_Preprocess(Name_df , ):
    Name_df = Name_df[[ 'name', 'BirthYear', 'FirstName', 'LastName', 'gender',
       'message', 'userID']]
    
    
    #將object type 復原回int
    Name_df['BirthYear'] = Name_df.BirthYear.apply(lambda x: int(x))
    Name_df['gender'] = Name_df.gender.apply(lambda x: int(float(x)))
    Name_df['message'] =  Name_df.message.apply(lambda x: int(x))

    #自動轉型
    Name_df , NA_list = reduce_mem_usage( Name_df)
    
    print("Name_df len",len(Name_df))
    Name_df = NamePreprecoess(Name_df)
    
    Totalname_list, moe_data_dict, common_dict, son_in_list, mu_in_list, special_word_dict = Read_Saved_Files()
 
    '''
    1 #用17G的維基百科+幾十M的萌典範文訓練的w2v, 每個entry是一個字
    '''
    
    Name_df , w2v_feature = add_W2V_feature( Name_df,w2v_Vector_number = 100)
    
    ################################################################
    '''
    2. 加入拼音feature  - one-hot encoding
    拼音是根據萌典與新華字典切出來的
    子音 = consonant = sonin 
    母音 = vowel = muin
    之前是隨意取名
    '''
    Name_df , phonetic_feature = add_phonetic_feature( Name_df)
    
    '''
    3. Fortune map - one-hot encoding
    '''
    Name_df , fortune_map_feature_list = add_fortune_map_feature ( Name_df )
    
    '''
    4. Character - one-hot encoding
    radical
    '''   
    Name_df , radical_feature_list = add_radical_feature ( Name_df )

    '''
    5. Zodiac_feature - one-hot encoding
    '''
    Name_df, Zodiac_feature_list = add_zodiac_feature( Name_df )
    
    return Name_df