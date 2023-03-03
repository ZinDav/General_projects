import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000):
    # Уберем самые популярные товары (их и так купят)
    popularity = pd.DataFrame(data.groupby('item_id')['user_id'].nunique() / data['user_id'].nunique()).reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data.loc[data['item_id'].isin(top_popular), 'item_id'] = 999999

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data.loc[data['item_id'].isin(top_notpopular), 'item_id'] = 999999

    # Уберем товары, которые не продавались за последние 12 месяцев
    day = data.groupby('item_id')['day'].max().reset_index()
    day = day[day['day'] < day['day'].max() - 365].item_id.tolist()
    data.loc[data['item_id'].isin(day), 'item_id'] = 999999

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    data['cost'] = data['sales_value'] / data['quantity']
    np.nan_to_num(data['cost'], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    costs = pd.DataFrame(data.groupby('item_id')['cost'].mean()).reset_index()
    low_cost = costs[costs['cost'] <= 0.5].item_id.tolist()

    data.loc[data['item_id'].isin(low_cost), 'item_id'] = 999999

    # Уберем слишком дорогие товары
    high_cost = costs[costs['cost'] > 100].item_id.tolist()
    data.loc[data['item_id'].isin(high_cost), 'item_id'] = 999999
    
    data = data.drop('cost', axis=1)

    # Top-N
    popularity = popularity[popularity['share_unique_users'] <= 0.5].sort_values('share_unique_users', ascending=False)
    top = popularity[popularity['item_id'].isin(data['item_id'])].item_id.head(take_n_popular).values
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    
    return data