import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier, plot_importance

from src.utils import prefilter_items
from src.recommenders import MainRecommender

from src.features import generate_fs

def tlvl_recommender(data, week_sep, item_features, user_features, N):
    """Создает двухуровневую систему рекомендаций
    
    --- 
    week_sep: количество недель в train_lvl_2,
    item_features: таблица с items, 
    user_features: таблица с users, 
    N: количество кандидатов
    ---
    Возвращает MainRecommender и 2-level предсказания
    """
    # Разделение датасета
    data_train_lvl_1 = data[data['week_no'] < data['week_no'].max() - week_sep]
    data_train_lvl_2 = data[(data['week_no'] >= data['week_no'].max() - week_sep)]

    # Фильтрация items
    data_train_lvl_1 = prefilter_items(data_train_lvl_1, take_n_popular=5000)
    
    # Инициализация и обучение MainRecommender
    recommender = MainRecommender(data_train_lvl_1, weighting='bm25', fake_id=999999)
    
    # 1 Уровень: Выборка users
    users_lvl_2 = pd.DataFrame(data_train_lvl_2['user_id'].unique())
    users_lvl_2.columns = ['user_id']

    train_users = data_train_lvl_1['user_id'].unique()
    users_lvl_2 = users_lvl_2[users_lvl_2['user_id'].isin(train_users)]

    # Генерация кандидатов
    users_lvl_2['candidates'] = users_lvl_2['user_id'].apply(
        lambda x: recommender.get_als_recommendations(x, N=N))
    
    # 2 Уровень: Разбиение кандидатов
    s = users_lvl_2.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'item_id'

    users_lvl_2 = users_lvl_2.drop('candidates', axis=1).join(s)
    users_lvl_2['flag'] = 1
    
    # Соединение рекомендаций и релевантных items
    targets_lvl_2 = data_train_lvl_2[['user_id', 'item_id']].copy()
    targets_lvl_2['target'] = 1

    targets_lvl_2 = users_lvl_2.merge(targets_lvl_2, on=['user_id', 'item_id'], how='left')

    targets_lvl_2['target'].fillna(0, inplace= True)
    targets_lvl_2.drop('flag', axis=1, inplace=True)
    
    # Создание фичей
    user_emb = pd.DataFrame(recommender.model.user_factors[:-10], columns=[f'users_{i}' for i in range(350)])

    pca = PCA(n_components=4, random_state=42)
    us_emb = pd.DataFrame(pca.fit_transform(user_emb))

    kmeans = KMeans(n_clusters=7, random_state=4, n_init=15)
    user_emb['user_cluster'] = kmeans.fit_predict(user_emb)

    user_emb['user_id'] = data_train_lvl_1.sort_values('user_id').user_id.unique()
    us_emb['user_id'] = data_train_lvl_1.sort_values('user_id').user_id.unique()

    user_emb = user_emb[['user_id', 'user_cluster']]
    user_emb = user_emb.merge(us_emb, on='user_id', how='inner')
    
    item_emb = pd.DataFrame(recommender.model.item_factors, columns=[f'item_{i}' for i in range(350)])

    pca = PCA(n_components=4, random_state=42)
    it_emb = pd.DataFrame(pca.fit_transform(item_emb))

    kmeans = KMeans(n_clusters=7, random_state=4, n_init=15)
    item_emb['item_cluster'] = kmeans.fit_predict(item_emb)

    item_emb['item_id'] = data_train_lvl_1.sort_values('item_id').item_id.unique()
    it_emb['item_id'] = data_train_lvl_1.sort_values('item_id').item_id.unique()

    item_emb = item_emb[['item_id', 'item_cluster']]
    item_emb = item_emb.merge(it_emb, on='item_id', how='inner')
    
    # Объединение targets_lvl_2 с фичами
    targets_lvl_2 = targets_lvl_2.merge(item_emb, on='item_id', how='left')
    targets_lvl_2 = targets_lvl_2.merge(user_emb, on='user_id', how='left')
    targets_lvl_2 = targets_lvl_2.merge(
        generate_fs(data_train_lvl_2, item_features, user_features), on=['user_id', 'item_id'], how='left')
    
    # Создание X_train и y_train
    X_train = targets_lvl_2.drop('target', axis=1)
    y_train = targets_lvl_2[['target']]
    
    cat_feats = ['manufacturer', 'department', 'brand', 'commodity_desc', 'sub_commodity_desc', 
                 'curr_size_of_product', 'age_desc', 'marital_status_code', 'income_desc', 
                 'homeowner_desc', 'hh_comp_desc', 'household_size_desc', 'kid_category_desc',
                'daytime', 'pop_cat', 'item_cluster', 'user_cluster']
    X_train[cat_feats] = X_train[cat_feats].astype('category')
    
    # Обучение классификатора
    lgb = LGBMClassifier(objective='binary', 
                         max_depth=5, 
                         n_estimators=75,
                         categorical_column=cat_feats)
    lgb.fit(X_train, y_train)

    # Предсказание
    X_train['proba'] = lgb.predict_proba(X_train)[:, 1]

    # Сортировка items
    res = pd.DataFrame(targets_lvl_2[targets_lvl_2['target'] == 1]\
                       .groupby('user_id')['item_id'].unique())\
                        .rename(columns={'item_id': 'actual'})

    res['items'] = X_train.groupby('user_id')['item_id'].agg(lambda x: [x.values])
    res['proba'] = X_train.groupby('user_id')['proba'].agg(lambda x: [x.values.argsort()[::-1]])
    res['pred'] = res.apply(lambda row: 
                                  pd.Series(row['items'][0][row['proba'][0]].flatten()).unique(), axis=1)
    res = res.reset_index()

    return recommender, res[['user_id', 'pred']]