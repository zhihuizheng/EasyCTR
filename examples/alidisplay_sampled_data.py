# coding: utf-8
import os
import gc
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder


def join_list(array, sep='-'):
    return sep.join(map(str, array))


def sampled(source_dir):
    user = pd.read_csv(os.path.join(source_dir, 'user_profile.csv'))
    sample = pd.read_csv(os.path.join(source_dir, 'raw_sample.csv'))

    if fraction < 1.0:
        user_sub = user.sample(frac=fraction, random_state=1024)
    else:
        user_sub = user
    sample_sub = sample.loc[sample.user.isin(user_sub.userid.unique())]

    behavior_log = pd.read_csv(os.path.join(source_dir, 'behavior_log.csv'))
    behavior_log = behavior_log.loc[behavior_log['btag'] == 'pv']

    userset = user_sub.userid.unique()
    behavior_log = behavior_log.loc[behavior_log.user.isin(userset)]

    ad = pd.read_csv(os.path.join(source_dir, 'ad_feature.csv'))
    ad['brand'] = ad['brand'].fillna(-1)

    # lbe = LabelEncoder()
    # unique_cate_id = np.concatenate((ad['cate_id'].unique(), behavior_log['cate'].unique()))
    #
    # lbe.fit(unique_cate_id)
    # ad['cate_id'] = lbe.transform(ad['cate_id']) + 1
    # behavior_log['cate'] = lbe.transform(behavior_log['cate']) + 1
    #
    # lbe = LabelEncoder()
    # # unique_brand = np.ad['brand'].unique()
    # # behavior_log = behavior_log.loc[behavior_log.brand.isin(unique_brand)]
    # unique_brand = np.concatenate((ad['brand'].unique(), behavior_log['brand'].unique()))
    #
    # lbe.fit(unique_brand)
    # ad['brand'] = lbe.transform(ad['brand']) + 1
    # behavior_log['brand'] = lbe.transform(behavior_log['brand']) + 1

    behavior_log = behavior_log.loc[behavior_log.user.isin(sample_sub.user.unique())]
    behavior_log.drop(columns=['btag'], inplace=True)
    behavior_log = behavior_log.loc[behavior_log['time_stamp'] > 0]

    return {
        'user': user_sub,
        'ad': ad,
        'examples': sample_sub,
        'behavior_log': behavior_log
    }


def gen_session_list_din(uid, t):
    t.sort_values('time_stamp', inplace=True, ascending=True)
    session_list = []
    session = []
    for row in t.iterrows():
        time_stamp = row[1]['time_stamp']
        cate_id = row[1]['cate']
        brand = row[1]['brand']
        session.append((cate_id, brand, time_stamp))

    if len(session) > 2:
        session_list.append(session[:])
    return uid, session_list


def applyParallel(df_grouped, func, n_jobs, backend='multiprocessing'):
    """Use Parallel and delayed """  # backend='threading'
    results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
        delayed(func)(name, group) for name, group in df_grouped)
    return {k: v for k, v in results}


# for din
def gen_user_hist_sessions(sampled_dict):
    user = sampled_dict['user']
    behavior_log = sampled_dict['behavior_log']
    behavior_log = behavior_log.loc[behavior_log.time_stamp >= 1493769600]  # 0503-0513
    # 0504~1493856000
    # 0503 1493769600

    n_samples = user.shape[0]
    print(n_samples)
    batch_size = 150000
    iters = (n_samples - 1) // batch_size + 1

    print("total", iters, "iters", "batch_size", batch_size)
    user_hist_session = {}
    for i in tqdm(range(0, iters)):
        target_user = user['userid'].values[i * batch_size:(i + 1) * batch_size]
        sub_data = behavior_log.loc[behavior_log.user.isin(target_user)]
        df_grouped = sub_data.groupby('user')
        user_hist_session_ = applyParallel(df_grouped, gen_session_list_din, n_jobs=20, backend='loky')
        user_hist_session.update(user_hist_session_)
    sampled_dict['user_hist_session'] = user_hist_session
    gc.collect()
    return sampled_dict


def gen_sess_feature_din(row, user_hist_session, max_seq_len):
    sess_input_dict = {'cate_id': [0], 'brand': [0]}
    sess_input_length = 0
    user, time_stamp = row[1]['user'], row[1]['time_stamp']
    if user not in user_hist_session or len(user_hist_session[user]) == 0:
        sess_input_dict['cate_id'] = join_list([])
        sess_input_dict['brand'] = join_list([])
        sess_input_length = 0
    else:
        cur_sess = user_hist_session[user][0]
        for i in reversed(range(len(cur_sess))):
            if cur_sess[i][2] < time_stamp:
                sess_input_dict['cate_id'] = join_list([e[0] for e in cur_sess[max(0, i + 1 - max_seq_len):i + 1]])
                sess_input_dict['brand'] = join_list([e[1] for e in cur_sess[max(0, i + 1 - max_seq_len):i + 1]])
                sess_input_length = len(sess_input_dict['brand'])
                break
    return sess_input_dict['cate_id'], sess_input_dict['brand'], sess_input_length


def gen_din_input(sampled_dict, max_seq_len):
    user_hist_session = sampled_dict['user_hist_session']
    sample_sub = sampled_dict['examples']

    sess_input_dict = {'cate_id': [], 'brand': []}
    sess_input_length = []
    for row in tqdm(sample_sub[['user', 'time_stamp']].iterrows()):
        a, b, c = gen_sess_feature_din(row, user_hist_session, max_seq_len)
        sess_input_dict['cate_id'].append(a)
        sess_input_dict['brand'].append(b)
        sess_input_length.append(c)
    sample_sub['hist_cate_id'] = sess_input_dict['cate_id']
    sample_sub['hist_brand'] = sess_input_dict['brand']

    user = sampled_dict['user']
    ad = sampled_dict['ad']
    user = user.fillna(-1)
    user.rename(columns={'new_user_class_level ': 'new_user_class_level'}, inplace=True)

    sample_sub = sampled_dict['examples']
    sample_sub.rename(columns={'user': 'userid'}, inplace=True)

    data = pd.merge(sample_sub, user, how='left', on='userid')
    data = pd.merge(data, ad, how='left', on='adgroup_id')
    return data


    # for feat in tqdm(sparse_features):
    #     lbe = LabelEncoder()  # or Hash
    #     data[feat] = lbe.fit_transform(data[feat])
    # mms = StandardScaler()
    # data[dense_features] = mms.fit_transform(data[dense_features])
    #
    # sparse_feature_list = [SingleFeat(feat, data[feat].max(
    # ) + 1) for feat in sparse_features + ['cate_id', 'brand']]
    #
    # dense_feature_list = [SingleFeat(feat, 1) for feat in dense_features]
    # sess_feature = ['cate_id', 'brand']
    #
    # sess_input = [pad_sequences(
    #     sess_input_dict[feat], maxlen=DIN_SESS_MAX_LEN, padding='post') for feat in sess_feature]

    # model_input = [data[feat.name].values for feat in sparse_feature_list] + \
    #               [data[feat.name].values for feat in dense_feature_list]
    # sess_lists = sess_input  # + [np.array(sess_input_length)]
    # model_input += sess_lists

    # pd.to_pickle(model_input, '../model_input/din_input_' +
    #              str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    # pd.to_pickle([np.array(sess_input_length)], '../model_input/din_input_len_' +
    #              str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    #
    # pd.to_pickle(data['clk'].values, '../model_input/din_label_' +
    #              str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl')
    # pd.to_pickle({'sparse': sparse_feature_list, 'dense': dense_feature_list},
    #              '../model_input/din_fd_' + str(FRAC) + '_' + str(DIN_SESS_MAX_LEN) + '.pkl', )

    # print("gen din input done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='../raw_data1', help='The source data directory.')
    parser.add_argument('--target_dir', type=str, default='../sampled_data', help='The target data directory.')
    parser.add_argument("--fraction", type=float, default=0.1, help='Sample fraction')
    parser.add_argument("--max_seq_len", type=int, default=10, help='Max sequence length')

    args = vars(parser.parse_args())
    source_dir = args['source_dir']
    target_dir = args['target_dir']
    fraction = args['fraction']
    max_seq_len = args['max_seq_len']

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    sampled_dict = sampled(source_dir)
    sampled_dict = gen_user_hist_sessions(sampled_dict)
    din_input = gen_din_input(sampled_dict, max_seq_len)

    # categorical_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
    #                         'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id',
    #                         'customer']
    # numeric_features = ['price']
    # sequence_features = []
    din_input.to_csv(os.path.join(target_dir, 'sampled_din.csv'), index=False)

    print("done!")
