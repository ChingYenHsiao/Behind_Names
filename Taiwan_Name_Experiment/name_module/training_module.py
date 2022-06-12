from itertools import chain, combinations
import datetime
import numpy as np
import random
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn import metrics

from name_module.preprocess import *


def get_all_combinations(feature_list):
    feature_combinations = [[x] for x in feature_list]
    for i in range(2, len(feature_list) + 1):
        for combo in combinations(feature_list, r=i):
            feature_combinations.append(list(combo))
    return feature_combinations


def _normalize(x, train=True, specified_column=None, x_mean=None, x_std=None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column is None:
        # np.arange(X.shape[1])只拿的到pd的shape  [0~n],不能拿到pd的欄位名稱
        specified_column = np.arange(x.shape[1])
        # specified_column = X.columns
    if train:
        # X_mean為一個list放每行的mean
        x_mean = np.mean(x[:, specified_column], 0).reshape(1, -1)
        x_std = np.std(x[:, specified_column], 0).reshape(1, -1)

    x[:, specified_column] = (x[:, specified_column] - x_mean) / (x_std + 1e-8)

    return x, x_mean, x_std


def w2v_normalize(sampled_df, w2v_feature):
    w2v_np = sampled_df[w2v_feature].to_numpy()
    # Normalize training and testing data
    w2v_np, X_mean, X_std = _normalize(w2v_np, train=True)
    idx = 0
    for i in range(100):
        for j in range(1, 3):
            col = "FN{}_wv_{}".format(j, i)
            FN_w2vX = []
            for vector_list in w2v_np:
                FN_w2vX.append(vector_list[idx])
            idx += 1
            sampled_df[col] = FN_w2vX
    return sampled_df


def name_gender_count(df):
    """ 用來扣掉同名字但有不同性別存在的名字
        name_gender_dict內大於0的就是男生,反之女生,0則男女各半

    Args:
        df (pd.DataFrame): name_df

    Returns:
        name_gender_dict(dict): dictionary with male score
    """
    name_gender_dict = {}
    for name, gender in zip(df.FirstName, df.gender):
        male_score = gender if gender == 1 else -1
        name_gender_dict[name] = name_gender_dict.get(name, 0) + male_score

    return name_gender_dict


def add_most_gender(name, name_gender_dict):
    if name_gender_dict[name] > 0:
        return 1
    elif name_gender_dict[name] < 0:
        return 0
    elif name_gender_dict[name] == 0:
        return -1


def first_name_augmentation(df):
    rename_dic = {}
    for col in df.columns:
        if "FN1" in col:
            rename_dic[col] = col.replace("FN1", "FN2")
        elif "FN2" in col:
            rename_dic[col] = col.replace("FN2", "FN1")
    df = pd.concat([df, df.rename(columns=rename_dic)], axis=0)

    return df.fillna(0)


def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param dataset: Pandas dataframe
    :return: None, print the basic statistics of the dataset
    """
    print(dataset.describe())


def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    # http://dataaspirant.com/2017/06/26/random-forest-classifier-python-scikit-learn/
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y


def random_forest_classifier(features, target, estimators_num=32,  min_samples_leaf_num=1):
    """
    To train the random forest classifier with features and target data
    :param min_samples_leaf_num:
    :param estimators_num:
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(
        n_estimators=estimators_num, n_jobs=4, min_samples_leaf=min_samples_leaf_num, random_state=42)
    print('estimators_num = ', estimators_num, 'min_samples_leaf_num = ',
          min_samples_leaf_num, "Training Data len = ", len(features))
    clf.fit(features, target)
    return clf


def random_forest_regressor(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestRegressor(n_estimators=32, n_jobs=4, min_samples_leaf=1)
    clf.fit(features, target)
    return clf


def RFC_metrics(data_x, data_y, trained_model):
    cf_matrix = confusion_matrix(data_y, trained_model.predict(data_x))
    TN = cf_matrix[0][0]
    FP = cf_matrix[0][1]
    FN = cf_matrix[1][0]
    TP = cf_matrix[1][1]
    precison = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TN + FP + FN + TP)
    F1 = 2 * precison * recall / (precison + recall)
    print("Accuracy :: ", round(accuracy, 4))
    print("Precision_score :: ", round(precison, 4))
    print("Recall_score :: ", round(recall, 4))
    print("F1_score :: ", round(F1, 4))
    return round(accuracy, 4), round(precison, 4), round(recall, 4),  round(F1, 4)


def multi_RFC_metrics(predicted, data_y):
    accuracy = accuracy_score(data_y, predicted)
    print("Accuracy :: ", accuracy)

    macro_precision = round(precision_score(
        data_y, predicted, average='macro'), 4)
    micro_precision = round(precision_score(
        data_y, predicted, average='micro'), 4)
    print("macro precision_score :: ", macro_precision)
    print("micro precision_score :: ", micro_precision)

    macro_recall = round(recall_score(data_y, predicted, average='macro'), 4)
    micro_recall = round(recall_score(data_y, predicted, average='micro'), 4)
    print("macro recall_score :: ", macro_recall)
    print("micro recall_score :: ", micro_recall)

    macro_f1 = round(f1_score(data_y, predicted, average='macro'), 4)
    micro_f1 = round(f1_score(data_y, predicted, average='micro'), 4)
    print("macro F1_score :: ", macro_f1)
    print("micro F1_score :: ", micro_f1)

    return (accuracy, macro_precision, micro_precision,
            macro_recall, micro_recall, macro_f1, micro_f1)


def add_gender_feature(name_df, gender_model, gender_x_feature):
    for feature in gender_x_feature:
        if feature not in name_df.columns:
            name_df[feature] = 0
    predict_proba = gender_model.predict_proba(name_df[gender_x_feature])
    male_prob_list = []
    female_prob_list = []
    for female_prob, male_prob in predict_proba:
        female_prob_list.append(female_prob)
        male_prob_list.append(male_prob)
    name_df['Male_prob'] = male_prob_list
    name_df['Female_prob'] = female_prob_list
    return name_df


def get_name_in_year_range_count(df_index, data_df, year_column):
    """ get name count in every year range dict.

    Args:
        df_index (index): index of selected sample
        data_df (pd.DataFrame): name dataframe
        year_column (str): 

    Returns:
        name_in_year_range_count ()
    """
    name_in_year_range_count = {}
    for index in df_index:
        first_name = data_df.loc[index, 'FirstName']
        year = data_df.loc[index, year_column]  # year = message / birthyear
        if first_name in name_in_year_range_count:
            name_in_year_range_count[first_name][year] = \
                name_in_year_range_count[first_name].get(year, 0) + 1
        else:
            name_in_year_range_count[first_name] = {year: 1}

    return name_in_year_range_count


def get_min_distance(year_range, answer_ranges, birth_year_base):
    smallest_dist = 999
    predict_year = birth_year_base + year_range * 5

    for answer_range in answer_ranges:
        answer_range = int(answer_range)
        if answer_range > year_range:
            closest_answer_year = predict_year + 4
        else:
            closest_answer_year = predict_year
        dist = abs(closest_answer_year - answer_range)
        if dist < smallest_dist:
            smallest_dist = dist
    return smallest_dist


def get_average_dist_error_and_multi_accuracy(df_index, predictions, sampled_df, birth_year_base):
    dist_error = 0
    multi_answer_accuracy = 0

    test_names = sampled_df.loc[df_index].FirstName.values
    name_in_year_range_count = get_name_in_year_range_count(
        df_index, sampled_df, 'BirthYear')
    name_in_year_count = get_name_in_year_range_count(
        df_index, sampled_df, 'message')

    print("dataset 有{}個名字 {}種名字".format(
        len(df_index), len(name_in_year_range_count)))

    for test_name, predict_year_range in zip(test_names, predictions):
        if test_name in name_in_year_range_count:
            if predict_year_range in name_in_year_range_count[test_name]:
                # 若test name 所有叫宜欣的年齡分布是   1=3個  2=6個  7=8個  9=1個人
                multi_answer_accuracy += 1
            else:
                # 計算錯誤，把平均絕對值誤差加上去
                dist_error += get_min_distance(
                    predict_year_range, name_in_year_count[test_name], birth_year_base)
    avg_dist_error = dist_error / len(test_names)
    multi_answer_accuracy = multi_answer_accuracy / len(test_names)
    return avg_dist_error, multi_answer_accuracy


def sample_name_df(df, sample_number, birth_years, ignore_index):
    """
    本來是用這段用到的BalanceCascade來sample,不過這函式不能用了,改用下面的方式來sample看看
    from imblearn.ensemble import BalanceCascade
    # Apply Balance Cascade method
    bc = BalanceCascade(return_indices = True , classifier = 'random-forest')
    X_resampled, y_resampled,indice = bc.fit_sample(train_compare_df[x_prob_feature],train_compare_df['compare_result'])
    """
    sampled_df = pd.DataFrame()
    for birth_year in birth_years:
        sampled_df = pd.concat([sampled_df, df[df["BirthYear"].apply(lambda x: x == birth_year)].sample(
            n=sample_number, frac=None, replace=True, weights=None, random_state=np.random.RandomState(), axis=0)],
            ignore_index=ignore_index)
    return sampled_df


def merge_birth_year(birth_year, head, tail):
    """  Merge head or tail data for unbalance dataset.
    Args:
        birth_year (int): birth year
        head (int): head birth year
        tail (int): tail birth year
    """
    if birth_year <= head:
        return head
    elif head < birth_year < tail:
        return birth_year
    elif tail <= birth_year:
        return tail


def train_birth_year_model(name_df, do_first_name_augmentation, validation_times,
                           feature_combinations, birth_year_base, target_names,
                           save_path, model_name, birth_years, sample_number):
    """ Train birth year model.

    Args:
        name_df (pd.DataFrame): name dataframe
        do_first_name_augmentation (bool): do first name augmentation or not.
        validation_times (int): validation times.
        feature_combinations (list): feature combinations for trying all feature combinations.
        birth_year_base (int): birth year base.
        target_names (list): target names for every birth year range.
        save_path (path): save model path.
        model_name (str): model name to save.
        birth_years (list): selected birth years range to train.
        sample_number (int): sample number for each birth year range.

    Returns:
        result_df (pd.DataFrame): result dataframe.
        saved_model (dict): saved model which has min dist error and multi answer accuracy.
        saved_feature (dict): saved feature.
    """
    result_all = {"Type": [], "feature": [], "lens": [], "accuracy": [], "multi_ans_accuracy": [],
                  "avg_dist": [], "macro_precision": [], "micro_precision": [],
                  "macro_recall": [], "micro_recall": [], "macro_F1": [], "micro_F1": []}

    y_feature = 'BirthYear'
    saved_feature, saved_feature_category, saved_model = None, None, None
    result_df = None
    min_distance = 100

    if not save_path.exists():
        os.mkdir(save_path)
        os.mkdir(save_path / "TrainedModel")
    elif not (save_path / "TrainedModel").exists():
        os.mkdir(save_path / "TrainedModel")

    for i, feature in enumerate(feature_combinations):
        sampled_df = sample_name_df(name_df, sample_number, birth_years, True)
        print(sampled_df.shape, len(birth_years))

        x_feature = get_x_feature(feature, sampled_df.columns)
        feature_category = ''.join([x[0].upper() for x in feature]).upper()
        print("Combination {} Training feature category: {}".format(i, feature))
        print("len of x_feature:", len(x_feature))

        for test_time in range(validation_times):
            dev_df = sampled_df.sample(
                n=len(sampled_df) // 10, frac=None, replace=False, weights=None, random_state=None, axis=0)
            if do_first_name_augmentation:
                train_x, test_x, train_y, test_y = split_dataset(first_name_augmentation(
                    sampled_df.drop(dev_df.index)), 0.7, x_feature, y_feature)
            else:
                train_x, test_x, train_y, test_y = split_dataset(
                    sampled_df.drop(dev_df.index), 0.7, x_feature, y_feature)
            # Create random forest classifier instance
            trained_model = random_forest_classifier(
                train_x, train_y.values.reshape(-1, 1).ravel(), estimators_num=64, min_samples_leaf_num=1)
            print('Finished training')

            for item in ("Train", "Test", "Development"):
                print("{} metrics".format(item))
                print("{}_x len = {}".format(item, len(train_x)))

                if item == "Train":
                    x_pred, y_true = train_x, train_y
                    df_index = train_x.index
                elif item == "Test":
                    x_pred, y_true = test_x, test_y
                    df_index = test_x.index
                else:
                    x_pred, y_true = dev_df[x_feature], dev_df[y_feature]
                    df_index = dev_df.index

                y_pred = trained_model.predict(x_pred)
                print('Finished prdeiction')
                accuracy, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1 = \
                    multi_RFC_metrics(y_pred, y_true)

                avg_dist_error, multi_answer_accuracy = get_average_dist_error_and_multi_accuracy(
                    df_index, y_pred, sampled_df, birth_year_base)
                print("Multi_{} Accuracy:: {}".format(
                    item, multi_answer_accuracy))
                print('平均年份絕對值誤差::', avg_dist_error)

                result_all["Type"].append(item)
                result_all["feature"].append(feature_category)
                result_all["lens"].append(len(x_feature))
                result_all["accuracy"].append(accuracy)
                result_all["multi_ans_accuracy"].append(multi_answer_accuracy)
                result_all["avg_dist"].append(avg_dist_error)
                result_all["macro_precision"].append(macro_precision)
                result_all["micro_precision"].append(micro_precision)
                result_all["macro_recall"].append(macro_recall)
                result_all["micro_recall"].append(micro_recall)
                result_all["macro_F1"].append(macro_f1)
                result_all["micro_F1"].append(micro_f1)

                # if item == "Development":
                print("report:\n", classification_report(
                    y_true, y_pred, target_names=target_names))

        result_df = pd.DataFrame(result_all)
        result_df.to_csv(save_path / 'result_all.csv', index=False)
        avg_dev_df = result_df[result_df["Type"] ==
                               "Development"].groupby(["feature"]).mean()
        avg_dev_df.to_csv(save_path / "average_development_result.csv")

        min_avg_dist_error = avg_dev_df["avg_dist"].min()
        if saved_model is None or min_distance > min_avg_dist_error:
            saved_model = trained_model
            min_distance = min_avg_dist_error
            saved_feature = x_feature
            saved_feature_category = feature_category

    if saved_model is not None:
        model_name = "{}_{}_model.pkl".format(
            model_name, saved_feature_category)
        feature_name = "{}_{}_feature.pkl".format(
            model_name, saved_feature_category)
        with open(save_path / "TrainedModel" / model_name, 'wb') as handle:
            pickle.dump(saved_model, handle)
        with open(save_path / "TrainedModel" / feature_name, 'wb') as handle:
            pickle.dump(saved_feature, handle)
        print("Output model Done.")
    return result_df, saved_model, saved_feature
