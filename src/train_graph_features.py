import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


# =========================
# CONFIG：這邊改成你的路徑就好
# =========================
DATA_FOLDER = os.getenv("TRANSACTION_DATA_DIR", "data")
TXN_PATH         = os.path.join(DATA_FOLDER, "acct_transaction.csv")
ALERT_PATH       = os.path.join(DATA_FOLDER, "acct_alert.csv")
PREDICT_PATH     = os.path.join(DATA_FOLDER, "acct_predict.csv")

# 指向您已重新命名的 'fx.csv' 檔案
FX_PATH          = os.path.join(DATA_FOLDER, "fx.csv")

# *** 重要 ***
# 因為您的 fx.csv 檔案沒有 'date' 欄位, 請將下面這行保持為 True
USE_FX_MEAN      = True 

OUTPUT_PATH      = os.path.join(DATA_FOLDER, "result_boost.csv")

VAL_RATIO        = 0.4   # 使用 40% 的 "欠採樣" 資料作為驗證集
MIN_POSITIVE_OUT = 500   # 最終輸出至少 500 筆
RANDOM_STATE     = 42
UNDERSAMPLE_RATIO = 20 # 負樣本:正樣本 的比例 (例如 20:1)


# =========================
# 小工具
# =========================
def _to_flag(series):
    s = series.astype(str).str.strip().str.lower()
    mapped = s.map({
        'y':1,'n':0,'yes':1,'no':0,'true':1,'false':0,'1':1,'0':0
    })
    mapped = pd.to_numeric(mapped, errors='coerce').fillna(0).astype(int)
    return mapped


def _load_fx(fx_path):
    if fx_path is None or not os.path.exists(fx_path):
        print("[INFO] 未提供 FX 檔案或路徑錯誤，跳過匯率轉換。")
        return None
    try:
        fx = pd.read_csv(fx_path)
        fx.columns = [c.strip().lower() for c in fx.columns]
        need_cols = {'ccy', 'rate_to_twd'}
        if not need_cols.issubset(set(fx.columns)):
            print(f"[WARN] FX 檔 {fx_path} 缺少 'ccy' 或 'rate_to_twd' 欄位，跳過匯率轉換。")
            return None
        fx['ccy'] = fx['ccy'].astype(str).str.upper().str.strip()
        fx['rate_to_twd'] = pd.to_numeric(fx['rate_to_twd'], errors='coerce')
        if 'date' in fx.columns:
            fx['date'] = pd.to_datetime(fx['date'], errors='coerce').dt.date
        print(f"[INFO] 成功載入 FX 檔案: {fx_path}")
        return fx
    except Exception as e:
        print(f"[ERROR] 讀取 FX 檔案時出錯: {e}")
        return None


def _apply_fx(df, fx, base_col='txn_amt', ccy_col='currency_type', date_col='txn_date'):
    """
    【修正 v7.0】:
    1. ccy_col 預設值已改為 'currency_type'
    2. 修正 merge 邏輯，正確對應 df[ccy_col] 和 fx['ccy']
    """
    if fx is None or ccy_col not in df.columns:
        print(f"[INFO] 未執行匯率轉換 (FX資料未載入或缺少'{ccy_col}'欄位)。")
        return df, base_col

    df = df.copy()
    df[base_col] = pd.to_numeric(df[base_col], errors='coerce').fillna(0)
    df[ccy_col] = df[ccy_col].astype(str).str.upper().str.strip()

    # 確保 TWD 匯率為 1
    if 'TWD' not in fx['ccy'].unique():
        if 'date' in fx.columns:
             # 如果有日期，為每個日期添加 TWD=1
            all_dates = fx['date'].unique()
            if len(all_dates) > 0:
                twd_df = pd.DataFrame({'date': all_dates, 'ccy': 'TWD', 'rate_to_twd': 1.0})
                fx = pd.concat([fx, twd_df], ignore_index=True)
            else:
                twd_row = pd.DataFrame([{'ccy': 'TWD', 'rate_to_twd': 1.0}])
                fx = pd.concat([fx, twd_row], ignore_index=True)
        else:
            twd_row = pd.DataFrame([{'ccy': 'TWD', 'rate_to_twd': 1.0}])
            fx = pd.concat([fx, twd_row], ignore_index=True)

    if 'date' in fx.columns and date_col in df.columns and not USE_FX_MEAN:
        print("[INFO] 正在使用日期進行匯率轉換...")
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df['_txn_date_norm'] = df[date_col].dt.date
        else:
            df['_txn_date_norm'] = pd.to_datetime(df[date_col], errors='coerce').dt.date

        merged = df.merge(
            fx.rename(columns={'date':'_fx_date'}),
            left_on=['_txn_date_norm', ccy_col],
            right_on=['_fx_date', 'ccy'],
            how='left'
        )
    else:
        print("[INFO] 正在使用平均匯率 (無日期) 進行轉換...")
        if 'date' in fx.columns:
            fx_mean = fx.groupby('ccy', as_index=False)['rate_to_twd'].mean()
        else:
            fx_mean = fx[['ccy','rate_to_twd']].drop_duplicates()
        
        # --- 這裡就是關鍵修正 ---
        # 使用 left_on 和 right_on 來對應不同的欄位名稱
        merged = df.merge(fx_mean, left_on=ccy_col, right_on='ccy', how='left')
        # ------------------------

    # 處理 TWD 和 未知幣別 (例如 ccy_col 是 'TWD' 但 fx 檔裡沒有 TWD)
    # 未知幣別也當作 1.0 (雖然這可能不是最好的策略，但比 NA 好)
    merged['rate_to_twd'] = merged['rate_to_twd'].fillna(1.0)
    merged['amt_twd'] = merged[base_col] * merged['rate_to_twd']

    # 清理合併時用的欄位
    drop_cols_post_merge = ['_txn_date_norm','_fx_date','rate_to_twd']
    if ccy_col != 'ccy': # 如果欄位名不同，就把 fx 來的 'ccy' 欄位刪除
        drop_cols_post_merge.append('ccy')
        
    for col in drop_cols_post_merge:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)
    
    print("[INFO] 已套用匯率轉換，使用 'amt_twd' 作為金額欄位。")
    return merged, 'amt_twd'


def load_data(txn_path, alert_path, predict_path):
    df_txn = pd.read_csv(txn_path)
    df_alert = pd.read_csv(alert_path)
    df_test  = pd.read_csv(predict_path)
    df_txn.columns = [c.strip() for c in df_txn.columns]
    return df_txn, df_alert, df_test


def _derive_time_cols(df):
    if 'txn_time' in df.columns:
        t = pd.to_datetime(df['txn_time'], format='%H:%M:%S', errors='coerce')
        df['hour'] = t.dt.hour
        df['minute'] = t.dt.minute
        df['is_night'] = ((df['hour'] <= 6) | (df['hour'] >= 22)).astype(int)
    else:
        df['hour'] = 0
        df['minute'] = 0
        df['is_night'] = 0
    return df


def _acct_type_table(df):
    df_from = df[['from_acct','from_acct_type']].rename(
        columns={'from_acct':'acct','from_acct_type':'type'}
    )
    df_to   = df[['to_acct','to_acct_type']].rename(
        columns={'to_acct':'acct','to_acct_type':'type'}
    )
    df_all  = pd.concat([df_from, df_to], ignore_index=True).dropna()
    esun_by_acct = df_all.groupby('acct')['type'].max().rename('is_esun')
    esun_by_acct = (esun_by_acct == 1).astype(int)
    return esun_by_acct.reset_index()


# =========================
# Graph / Network 特徵
# =========================
def _graph_features(df, alert_set):
    edges = df[['from_acct','to_acct']].dropna().copy()
    edges['from_acct'] = edges['from_acct'].astype(str)
    edges['to_acct']   = edges['to_acct'].astype(str)

    out_deg = (
        edges.groupby('from_acct')['to_acct']
        .nunique()
        .rename('deg_out')
        .reset_index()
    )
    in_deg  = (
        edges.groupby('to_acct')['from_acct']
        .nunique()
        .rename('deg_in')
        .reset_index()
    )

    deg = pd.merge(out_deg, in_deg,
                   left_on='from_acct', right_on='to_acct', how='outer')
    deg['acct'] = deg['from_acct'].fillna(deg['to_acct'])
    deg = deg.drop(columns=['from_acct','to_acct'])
    deg['deg_out'] = deg['deg_out'].fillna(0)
    deg['deg_in']  = deg['deg_in'].fillna(0)
    deg['deg_total'] = deg['deg_out'] + deg['deg_in']

    neighbor_list = edges.groupby('from_acct')['to_acct'].apply(set).to_dict()

    direct_alert_prop = {}
    total_neighbor_cnt = {}
    twohop_alert_prop = {}
    twohop_cnt_all = {}

    for acct, nbrs in neighbor_list.items():
        nbrs = set(nbrs)
        total_neighbor_cnt[acct] = len(nbrs)

        alert_cnt = sum((n in alert_set) for n in nbrs)
        direct_alert_prop[acct] = (alert_cnt / len(nbrs)) if len(nbrs) else 0.0

        twohop = set()
        for n in nbrs:
            twohop |= neighbor_list.get(n, set())
        twohop.discard(acct)

        twohop_cnt_all[acct] = len(twohop)
        if len(twohop) == 0:
            twohop_alert_prop[acct] = 0.0
        else:
            alert2_cnt = sum((n in alert_set) for n in twohop)
            twohop_alert_prop[acct] = alert2_cnt / len(twohop)

    graph_df = pd.DataFrame({
        'acct': list(set(list(neighbor_list.keys()) + list(deg['acct'].astype(str))))
    })

    graph_df = graph_df.merge(deg, on='acct', how='left')
    graph_df['deg_out']   = graph_df['deg_out'].fillna(0)
    graph_df['deg_in']    = graph_df['deg_in'].fillna(0)
    graph_df['deg_total'] = graph_df['deg_total'].fillna(0)

    graph_df['nbr_alert_prop'] = graph_df['acct'].map(direct_alert_prop).fillna(0.0)
    graph_df['nbr_cnt']        = graph_df['acct'].map(total_neighbor_cnt).fillna(0).astype(float)

    graph_df['twohop_alert_prop'] = graph_df['acct'].map(twohop_alert_prop).fillna(0.0)
    graph_df['twohop_cnt']        = graph_df['acct'].map(twohop_cnt_all).fillna(0).astype(float)

    return graph_df


# =========================
# 特徵工程
# =========================
def build_features(df_txn, df_alert, fx=None):
    df = df_txn.copy()

    if 'txn_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['txn_date']):
        df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')

    df = _derive_time_cols(df)

    if 'txn_amt' in df.columns:
        df['txn_amt'] = pd.to_numeric(df['txn_amt'], errors='coerce').fillna(0)
    amt_col = 'txn_amt'

    if fx is not None:
        # 使用修正後的 _apply_fx，它知道 txn_data 上的欄位是 'currency_type'
        df, amt_col = _apply_fx(df, fx, base_col=amt_col,
                                ccy_col='currency_type', date_col='txn_date')

    if 'is_self_txn' in df.columns:
        df['is_self_txn_flag'] = _to_flag(df['is_self_txn'])
    else:
        df['is_self_txn_flag'] = 0

    send_grp = df.groupby('from_acct')
    recv_grp = df.groupby('to_acct')

    feat = pd.DataFrame({'acct': pd.Index(sorted(set(df['from_acct']) | set(df['to_acct'])))})

    if 'txn_date' in df.columns:
        date_stats_send = df.groupby('from_acct')['txn_date'].agg(['min','max']).rename(
            columns={'min':'send_min_date','max':'send_max_date'}
        )
        date_stats_recv = df.groupby('to_acct')['txn_date'].agg(['min','max']).rename(
            columns={'min':'recv_min_date','max':'recv_max_date'}
        )
        overall_max_date = df['txn_date'].max()
    else:
        date_stats_send = pd.DataFrame(columns=['send_min_date','send_max_date'],
                                       index=pd.Index([], name='from_acct'))
        date_stats_recv = pd.DataFrame(columns=['recv_min_date','recv_max_date'],
                                       index=pd.Index([], name='to_acct'))
        overall_max_date = pd.Timestamp.now()

    feat = feat.merge(date_stats_send, left_on='acct', right_index=True, how='left')
    feat = feat.merge(date_stats_recv, left_on='acct', right_index=True, how='left')

    feat['acct_min_date'] = feat[['send_min_date','recv_min_date']].min(axis=1)
    feat['acct_max_date'] = feat[['send_max_date','recv_max_date']].max(axis=1)

    if not feat['acct_max_date'].isnull().all():
        feat['acct_age_days']     = (feat['acct_max_date'] - feat['acct_min_date']).dt.days
        feat['acct_recency_days'] = (overall_max_date - feat['acct_max_date']).dt.days
    else:
        feat['acct_age_days']     = np.nan
        feat['acct_recency_days'] = np.nan

    def _add_prefixed(agg_df, key_col, prefix):
        return agg_df.add_prefix(prefix).rename_axis(key_col).reset_index()

    n_send = _add_prefixed(send_grp.size().to_frame('n'), 'from_acct', 'send_')
    n_recv = _add_prefixed(recv_grp.size().to_frame('n'), 'to_acct',   'recv_')

    amt_aggs = ['sum','mean','max','std','median']
    send_amt = _add_prefixed(send_grp[amt_col].agg(amt_aggs), 'from_acct', 'send_amt_')
    recv_amt = _add_prefixed(recv_grp[amt_col].agg(amt_aggs), 'to_acct',   'recv_amt_')

    for c in [c for c in send_amt.columns if 'std' in c]:
        send_amt[c] = send_amt[c].fillna(0)
    for c in [c for c in recv_amt.columns if 'std' in c]:
        recv_amt[c] = recv_amt[c].fillna(0)

    uniq_to   = _add_prefixed(send_grp['to_acct'].nunique().to_frame('nunique'),
                              'from_acct', 'uniq_to_')
    uniq_from = _add_prefixed(recv_grp['from_acct'].nunique().to_frame('nunique'),
                              'to_acct',   'uniq_from_')

    if 'hour' in df.columns:
        burst = (
            df.assign(hour_block=df['hour'])
              .groupby(['from_acct','hour_block']).size()
              .groupby('from_acct').max()
              .rename('send_burst_max')
              .reset_index()
        )
    else:
        burst = pd.DataFrame({'from_acct':[], 'send_burst_max':[]})

    night_rate = _add_prefixed(send_grp['is_night'].mean().rename('night_rate'),
                               'from_acct', 'send_')

    df['is_round_1000'] = (df[amt_col] % 1000 == 0).astype(int)
    df['is_round_100']  = (df[amt_col] % 100  == 0).astype(int)

    round1000 = _add_prefixed(send_grp['is_round_1000'].mean().rename('round1000'),
                              'from_acct','send_')
    round100  = _add_prefixed(send_grp['is_round_100'].mean().rename('round100'),
                              'from_acct','send_')

    self_rate = _add_prefixed(send_grp['is_self_txn_flag'].mean().rename('self_rate'),
                              'from_acct','send_')

    if 'channel_type' in df.columns:
        ch_counts = df.pivot_table(index='from_acct',
                                   columns='channel_type',
                                   values=amt_col,
                                   aggfunc='size',
                                   fill_value=0)
        ch_frac = ch_counts.div(
            ch_counts.sum(axis=1).replace(0,1),
            axis=0
        )
        ch_frac.columns = [f'send_ch_frac_{str(c)}' for c in ch_frac.columns]
        ch_feat = ch_frac.reset_index()
    else:
        ch_feat = pd.DataFrame({'from_acct':[],})

    alert_set = set(df_alert['acct'].astype(str).tolist())
    df['from_is_alert_neighbor'] = df['to_acct'].astype(str).isin(alert_set).astype(int)
    df['to_is_alert_neighbor']   = df['from_acct'].astype(str).isin(alert_set).astype(int)

    alert_to_send = _add_prefixed(
        send_grp['from_is_alert_neighbor'].mean().rename('alert_neighbor_rate'),
        'from_acct','send_'
    )
    alert_to_recv = _add_prefixed(
        recv_grp['to_is_alert_neighbor'].mean().rename('alert_neighbor_rate'),
        'to_acct','recv_'
    )

    mask_send = df['from_is_alert_neighbor'] == 1
    send_alert_amt_series = df.loc[mask_send].groupby('from_acct')[amt_col].sum()
    alert_amt_send = _add_prefixed(
        send_alert_amt_series.rename('alert_neighbor_amt'),
        'from_acct','send_'
    )

    mask_recv = df['to_is_alert_neighbor'] == 1
    recv_alert_amt_series = df.loc[mask_recv].groupby('to_acct')[amt_col].sum()
    alert_amt_recv = _add_prefixed(
        recv_alert_amt_series.rename('alert_neighbor_amt'),
        'to_acct','recv_'
    )

    def _safe_merge(left, right, left_key, right_key):
        return left.merge(
            right,
            left_on=left_key,
            right_on=right_key,
            how='left'
        ).drop(columns=[right_key], errors='ignore')

    feat = _safe_merge(feat, n_send,        'acct', 'from_acct')
    feat = _safe_merge(feat, n_recv,        'acct', 'to_acct')
    feat = _safe_merge(feat, send_amt,      'acct', 'from_acct')
    feat = _safe_merge(feat, recv_amt,      'acct', 'to_acct')
    feat = _safe_merge(feat, uniq_to,       'acct', 'from_acct')
    
    # --- 【已修正的 Typo】 ---
    feat = _safe_merge(feat, uniq_from,     'acct', 'to_acct')
    # --------------------------

    feat = _safe_merge(feat, burst,         'acct', 'from_acct')
    feat = _safe_merge(feat, night_rate,    'acct', 'from_acct')
    feat = _safe_merge(feat, round1000,     'acct', 'from_acct')
    feat = _safe_merge(feat, round100,      'acct', 'from_acct')
    feat = _safe_merge(feat, self_rate,     'acct', 'from_acct')
    if not ch_feat.empty:
        feat = _safe_merge(feat, ch_feat,   'acct', 'from_acct')
    feat = _safe_merge(feat, alert_to_send, 'acct', 'from_acct')
    feat = _safe_merge(feat, alert_to_recv, 'acct', 'to_acct')
    feat = _safe_merge(feat, alert_amt_send,'acct', 'from_acct')
    feat = _safe_merge(feat, alert_amt_recv,'acct', 'to_acct')

    feat['amt_in_out_ratio']  = feat['recv_amt_sum'] / (feat['send_amt_sum'].replace(0,1))
    feat['cnt_in_out_ratio']  = feat['recv_n']       / (feat['send_n'].replace(0,1))
    feat['uniq_in_out_ratio'] = feat['uniq_from_nunique'] / (feat['uniq_to_nunique'].replace(0,1))

    feat['net_inflow_sum']    = feat['recv_amt_sum'] - feat['send_amt_sum']

    feat['send_per_day']      = feat['send_n'] / (feat['acct_age_days'].replace(0,1))
    feat['recv_per_day']      = feat['recv_n'] / (feat['acct_age_days'].replace(0,1))
    feat['send_amt_per_day']  = feat['send_amt_sum'] / (feat['acct_age_days'].replace(0,1))
    feat['recv_amt_per_day']  = feat['recv_amt_sum'] / (feat['acct_age_days'].replace(0,1))

    esun_tbl = _acct_type_table(df)
    graph_df = _graph_features(df, alert_set)
    feat = feat.merge(graph_df, on='acct', how='left')

    def _log1p_safe(s):
        return np.log1p(s.clip(lower=0).fillna(0))

    log_cols = [
        'send_amt_sum', 'recv_amt_sum', 'send_burst_max', 'send_per_day',
        'recv_per_day', 'send_amt_per_day', 'recv_amt_per_day',
        'net_inflow_sum', 'deg_total', 'nbr_cnt', 'twohop_cnt',
        'send_amt_mean', 'send_amt_max', 'send_amt_median',
        'recv_amt_mean', 'recv_amt_max', 'recv_amt_median',
        'send_alert_neighbor_amt', 'recv_alert_neighbor_amt'
    ]
    
    for col in log_cols:
        if col in feat.columns:
            feat[f'log_{col}'] = _log1p_safe(feat[col])

    feat = feat.fillna(0)

    feat = feat.merge(esun_tbl, on='acct', how='left').fillna({'is_esun':0})
    feat['is_esun'] = feat['is_esun'].astype(int)

    date_cols = [
        'send_min_date','send_max_date','recv_min_date','recv_max_date',
        'acct_min_date','acct_max_date'
    ]
    for col in date_cols:
        if col in feat.columns:
            feat[col] = pd.to_datetime(feat[col], errors='coerce')
            feat[col] = feat[col].fillna(pd.Timestamp('1970-01-01'))

    return feat


# =========================
# train / test split
# =========================
def split_train_test(feat, df_alert, df_test):
    test_accts = set(df_test['acct'].astype(str))
    feat['acct_str'] = feat['acct'].astype(str)

    train_df = feat[(~feat['acct_str'].isin(test_accts)) & (feat['is_esun']==1)].copy()
    test_df  = feat[feat['acct_str'].isin(test_accts)].copy()

    y_train = train_df['acct_str'].isin(set(df_alert['acct'].astype(str))).astype(int)
    return train_df, y_train, test_df


# =========================
# 時間切分驗證 & 欠採樣
# =========================

def _undersample(train_df, y_train, negative_ratio=10, random_state=42):
    """
    對 train_df 和 y_train 進行隨機欠採樣 (Random Under-Sampling)。
    保留所有正樣本，並根據 negative_ratio 抽取負樣本。
    """
    print(f"[INFO] 進行欠採樣，負:正 比例 = {negative_ratio}:1")

    # 找出正負樣本的索引
    pos_indices = y_train[y_train == 1].index
    neg_indices = y_train[y_train == 0].index

    n_positive = len(pos_indices)
    
    if n_positive == 0:
        print("[WARN] 訓練數據中沒有正樣本，無法進行欠採樣。")
        return train_df.loc[neg_indices], y_train.loc[neg_indices]

    n_negative_keep = n_positive * negative_ratio

    if n_negative_keep > len(neg_indices):
        print(f"[WARN] 想要的負樣本數 ({n_negative_keep}) > 總負樣本數 ({len(neg_indices)})。使用所有負樣本。")
        n_negative_keep = len(neg_indices)
    
    # 隨機抽取負樣本
    rng = np.random.RandomState(random_state)
    neg_indices_sampled = rng.choice(neg_indices, size=int(n_negative_keep), replace=False)
    
    # 合併索引並打亂
    keep_indices = np.concatenate([pos_indices, neg_indices_sampled])
    rng.shuffle(keep_indices) # 使用 rng 進行打亂
    
    train_df_sampled = train_df.loc[keep_indices].copy()
    y_train_sampled = y_train.loc[keep_indices].copy()
    
    print(f"[INFO] 欠採樣完成：正樣本={n_positive}, 負樣本={len(neg_indices_sampled)}, 總計={len(train_df_sampled)}")
    
    return train_df_sampled, y_train_sampled


def _time_split_for_cv(train_df, y_train, features, val_ratio=0.25):
    tmp = train_df.copy()
    tmp['y'] = y_train.values

    if 'acct_max_date' not in tmp.columns or tmp['acct_max_date'].isnull().all():
        print("[WARN] 'acct_max_date' 不存在或全為空，使用隨機切分替代時間切分。")
        X_tr, X_val, y_tr, y_val = train_test_split(
            train_df[features], y_train, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_train
        )
    else:
        # 確保 acct_max_date 是可用於排序的格式
        tmp['acct_max_date'] = pd.to_datetime(tmp['acct_max_date'], errors='coerce').fillna(pd.Timestamp.min)
        tmp.sort_values('acct_max_date', inplace=True)
        
        val_size = int(len(tmp) * val_ratio)
        split_at = len(tmp) - val_size
        tr_df = tmp.iloc[:split_at]
        val_df= tmp.iloc[split_at:]

        X_tr  = tr_df[features]
        y_tr  = tr_df['y']
        X_val = val_df[features]
        y_val = val_df['y']
    
    print(f"[INFO] CV 切分: 訓練集 {len(X_tr)} 筆 (正樣本: {y_tr.sum()}), 驗證集 {len(X_val)} 筆 (正樣本: {y_val.sum()})")
    
    # 檢查驗證集中是否有正樣本
    if y_val.sum() == 0:
        print("[WARN] 警告：驗證集中沒有正樣本！F1-score 將為 0。")

    return X_tr, y_tr, X_val, y_val


def _find_best_threshold(y_true, y_proba):
    """
    在驗證集上暴力搜索最佳 F1 門檻值
    """
    best_f1 = 0.0
    best_thresh = 0.5 # 預設值
    
    # 從 0.01 到 0.99，間隔 0.01
    thresholds = np.linspace(0.01, 0.99, 99)
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh, best_f1


def _try_params_and_find_best_f1(X_tr, y_tr, X_val, y_val,
                                 params_list,
                                 base_params_extra=None):
    """
    新策略：
    1. 遍歷所有參數組合
    2. 使用 'roc_auc' 作為 early_stopping 的指標，找到最佳模型
    3. 用這個最佳模型預測驗證集機率
    4. 暴力搜索這個機率的最佳 F1 門檻值
    5. 記錄 F1、Prec、Rec 和最佳門檻
    6. 返回 F1 最高的模型和設定
    """
    best_f1 = -1
    best_info = None

    if y_val.sum() == 0:
        print("[ERROR] 驗證集中沒有正樣本，無法進行調優。請檢查 'VAL_RATIO' 或資料切分邏輯。")
        return None

    for pset in params_list:
        full_params = dict(base_params_extra)
        full_params.update(pset)

        clf = lgb.LGBMClassifier(**full_params)
        
        print(f"\n[INFO] G 正在測試參數: {pset}")
        
        clf.fit(X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='auc', # 使用 AUC
                callbacks=[lgb.early_stopping(100, verbose=False)])

        # 用最佳迭代次數的模型來預測機率
        best_iter = clf.best_iteration_ if (clf.best_iteration_ is not None and clf.best_iteration_ > 0) else full_params.get('n_estimators', 100)
        proba_val = clf.predict_proba(X_val, num_iteration=best_iter)[:,1]
        
        auc_score = roc_auc_score(y_val, proba_val)
        
        # 尋找這個模型對應的最佳 F1 門檻值
        best_thresh, f1 = _find_best_threshold(y_val, proba_val)
        
        # 計算該門檻值下的 P/R
        y_hat_best = (proba_val >= best_thresh).astype(int)
        prec, rec, _, _ = precision_recall_fscore_support(
            y_val, y_hat_best, average='binary', zero_division=0
        )

        print(f"[INFO] 測試結果 - AUC: {auc_score:.4f}, 最佳 F1: {f1:.4f} @ 門檻={best_thresh:.2f}")

        if f1 > best_f1:
            best_f1 = f1
            best_info = {
                'model': clf,
                'params': full_params,
                'thresh': best_thresh, # 儲存這個模型對應的最佳門檻
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'best_iter': best_iter
            }

    return best_info


# (移除了 post_rules_for_eval 和 post_rules_infer)


# =========================
# PIPELINE
# =========================
def run_pipeline():
    # 修正 t0 bug
    t0 = time.time()
    
    for p in [TXN_PATH, ALERT_PATH, PREDICT_PATH]:
        if not os.path.exists(p):
            print(f"[ERROR] 找不到檔案：{p}。請檢查 CONFIG 中的路徑設定。")
            return

    # --- 載入資料 ---
    t_load_start = time.time()
    print("[INFO] 載入資料 ...")
    fx_data = _load_fx(FX_PATH)
    df_txn, df_alert, df_test = load_data(TXN_PATH, ALERT_PATH, PREDICT_PATH)

    if 'txn_date' in df_txn.columns:
        df_txn['txn_date'] = pd.to_datetime(df_txn['txn_date'], errors='coerce')
    else:
        print("[WARN] 沒有 txn_date，時間特徵/時間切分會變弱")
    print(f"[INFO] 資料載入完成 (耗時 {time.time() - t_load_start:.2f}s)")

    # --- 特徵工程 ---
    t_feat_start = time.time()
    print("[INFO] 建立特徵 ...")
    feat = build_features(df_txn, df_alert, fx=fx_data)
    print(f"[INFO] 特徵工程完成 (耗時 {time.time() - t_feat_start:.2f}s)")

    # --- 資料切分 ---
    print("[INFO] 切 train/test ...")
    train_df, y_train, test_df = split_train_test(feat, df_alert, df_test)
    print(f"[INFO] S原始訓練集: {len(train_df)} 筆 (正樣本: {y_train.sum()})")
    print(f"[INFO] 測試集: {len(test_df)} 筆")

    # --- 【關鍵】進行欠採樣 ---
    train_df_sampled, y_train_sampled = _undersample(
        train_df, y_train, negative_ratio=UNDERSAMPLE_RATIO, random_state=RANDOM_STATE
    )

    drop_cols = {
        'acct','acct_str','is_esun',
        'send_min_date','send_max_date','recv_min_date','recv_max_date',
        'acct_min_date','acct_max_date'
    }

    features = [
        c for c in train_df.columns
        if c not in drop_cols and (np.issubdtype(train_df[c].dtype, np.number))
    ]
    print(f"[INFO] 使用 {len(features)} 個特徵進行訓練。")

    # --- 交叉驗證 ---
    print("[INFO] 時間切分驗證 (使用欠採樣資料)...")
    X_tr, y_tr, X_val, y_val = _time_split_for_cv(
        train_df_sampled, y_train_sampled, features, val_ratio=VAL_RATIO
    )
    
    # --- 【關鍵】移除模型內建的平衡參數 ---
    base_params_extra = dict(
        objective='binary',
        n_estimators=1000, 
        learning_rate=0.02,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1  # 關閉 LightGBM 的囉嗦日誌
    )

    param_grid = [
        dict(num_leaves=31, max_depth=-1, subsample=0.8, colsample_bytree=0.8),
        dict(num_leaves=63, max_depth=-1, subsample=0.8, colsample_bytree=0.8),
        dict(num_leaves=31, max_depth=8,  subsample=0.9, colsample_bytree=0.8),
        dict(num_leaves=63, max_depth=8,  subsample=0.9, colsample_bytree=0.8),
        dict(num_leaves=127, max_depth=-1, subsample=0.7, colsample_bytree=0.7),
        dict(num_leaves=63, max_depth=-1, subsample=0.7, colsample_bytree=0.7, min_child_samples=50),
        dict(num_leaves=31, max_depth=-1, subsample=0.9, colsample_bytree=0.9, min_child_samples=100),
        dict(num_leaves=90, max_depth=10, subsample=0.8, colsample_bytree=0.8),
    ]

    print("[INFO] 掃參數/threshold ...")
    t_search_start = time.time()
    
    # --- 【關鍵】使用新的調參函數 ---
    best_info = _try_params_and_find_best_f1(
        X_tr, y_tr, X_val, y_val,
        params_list=param_grid,
        base_params_extra=base_params_extra
    )
    # --------------------------------
    
    print(f"[INFO] 參數搜索完成 (耗時 {time.time() - t_search_start:.2f}s)")

    if best_info is None:
        print("[ERROR] 參數搜索失敗，未找到有效模型 (可能是驗證集無正樣本)。")
        return

    best_model       = best_info['model']
    best_thresh      = best_info['thresh']
    best_params      = best_info['params']
    best_f1          = best_info['f1']
    best_prec        = best_info['precision']
    best_rec         = best_info['recall']
    best_iter        = best_info.get('best_iter', base_params_extra['n_estimators'])

    print("======================================================")
    print(f"[CV-RESULT] 最佳 F1: {best_f1:.4f}")
    print(f"[CV-RESULT] 最佳 Precision: {best_prec:.4f}")
    print(f"[CV-RESULT] 最佳 Recall: {best_rec:.4f}")
    print(f"[CV-RESULT] S最佳 Thresh: {best_thresh:.4f}")
    print(f"[CV-RESULT] 最佳 Params: {best_params}")
    print(f"[CV-RESULT] 最佳 Iteration: {best_iter}")
    
    proba_val_best = best_model.predict_proba(X_val[features])[:,1]
    y_hat_val_best = (proba_val_best >= best_thresh).astype(int) # <-- 使用單一最佳門檻
    print("[CV-REPORT] (驗證集上的最終報告)\n", classification_report(y_val, y_hat_val_best, digits=4))
    print("======================================================")

    # --- 最終訓練 ---
    print("[INFO] 用最佳參數重訓 (使用 *全部* 欠採樣資料) ...")
    final_params = best_params.copy()
    
    final_params['n_estimators'] = best_iter if best_iter != -1 else base_params_extra['n_estimators']
    
    final_model = lgb.LGBMClassifier(**final_params)
    
    final_model.fit(train_df_sampled[features], y_train_sampled)

    print("[INFO] 對測試資料預測 ...")
    proba_test = final_model.predict_proba(test_df[features])[:,1]
    
    # --- 【關鍵】使用 CV 找到的最佳門檻值 ---
    y_pred_raw = (proba_test >= best_thresh).astype(int)
    # --------------------------------------

    # --- 後處理：保底 500 筆 ---
    pos_idx = np.where(y_pred_raw == 1)[0]
    num_pos = len(pos_idx)
    
    if num_pos < MIN_POSITIVE_OUT:
        print(f"[POST] 模型找到 {num_pos} positives < MIN_POSITIVE_OUT={MIN_POSITIVE_OUT}, 強制補高機率 top-K")
        # 拿模型預測的原始機率來排序
        top_idx = np.argsort(-proba_test)[:MIN_POSITIVE_OUT]
        y_pred_final = np.zeros_like(y_pred_raw)
        y_pred_final[top_idx] = 1
    else:
        y_pred_final = y_pred_raw
        print(f"[POST] 模型+門檻 找到 {num_pos} positives (>= {MIN_POSITIVE_OUT})，無需保底。")

    out = pd.DataFrame({
        'acct': test_df['acct'].values,
        'label': y_pred_final.astype(int)
    })

    # 確保所有測試集帳戶都在輸出中
    out = df_test[['acct']].merge(out, on='acct', how='left') \
                           .fillna({'label':0}) \
                           .astype({'label':int})

    out.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print("======================================================")
    print(f"[OK] Saved submission to {OUTPUT_PATH}")
    print(f"Predicted positives: {int(out['label'].sum())} / {len(out)}")
    
    t1 = time.time()
    print(f"[TIME] total pipeline: {t1 - t0:.2f}s")
    print("======================================================")


if __name__ == "__main__":
    run_pipeline()