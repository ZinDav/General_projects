import pandas as pd
import numpy as np


def generate_fs(data, item_features, user_features):
    data.sort_values(['week_no', 'day', 'trans_time'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Цена (value)
    data['value'] = (data['sales_value'] - data['retail_disc']) / data['quantity']
    np.nan_to_num(data['value'], copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # средняя цена в категории (value_cat_mean)
    data = data.merge(item_features, on='item_id', how='left')
    data = data.merge(data.groupby('department')['value'].mean(), on='department', 
                      how='left', suffixes=(None, '_cat_mean'))
    
    # среднее количество покупок 1 товара в категории в неделю (mean_pw)
    hd = data.groupby(['department', 'week_no'])['quantity'].mean().reset_index()\
                        .rename(columns={'quantity': 'mean_pw'})
    data = data.merge(hd, on=['department', 'week_no'], how='left')
    
    # Средний чек (mean_check)
    hd = data.groupby(['user_id', 'basket_id'])['sales_value'].sum().reset_index()\
                        .rename(columns={'sales_value': 'mean_check'})
    hd = hd.groupby('user_id')['mean_check'].mean()
    data = data.merge(hd, on='user_id', how='inner')

    # Доля покупок в каждой категории (us_cat_q)
    hd = (data.groupby(['user_id', 'department'])['quantity'].sum() / \
                        data.groupby('user_id')['quantity'].sum()).reset_index()\
                        .rename(columns={'quantity': 'us_cat_q'})
    data = data.merge(hd, on=['user_id', 'department'], how='left')
    
    # Долю покупок утром/днем/вечером (daytime_quant)
    data['daytime'] = np.nan
    data.loc[data['trans_time'] < 600, 'daytime'] = 'N'
    data.loc[(data['trans_time'] >= 600) & (data['trans_time'] < 1200), 'daytime'] = 'M'
    data.loc[(data['trans_time'] >= 1200) & (data['trans_time'] < 1800), 'daytime'] = 'D'
    data.loc[data['trans_time'] >= 1800, 'daytime'] = 'E'

    hd = (data.groupby(['user_id', 'daytime'])['quantity'].sum() / \
        data.groupby(['user_id'])['quantity'].sum()).reset_index()\
                        .rename(columns={'quantity': 'daytime_quant'})
    data = data.merge(hd, on=['user_id', 'daytime'], how='left')
    
    # Частотность покупок раз/месяц (cnt_pm)
    m = 1
    month = []
    for d in data['day'].unique():
        if (d - (m-1)*30.417)/30.417 > 1:
            m += 1
        month.append(m)

    hd = pd.DataFrame(data['day'].unique(), columns=['day'])
    hd['month'] = month
    data = data.merge(hd, on='day', how='inner')
    
#     data = data.reset_index()
    hd = data.groupby(['user_id', 'month'])['basket_id'].count().reset_index()\
                .rename(columns={'basket_id': 'cnt_pm'})
    data = data.merge(hd, on=['user_id', 'month'], how='left')
    
    # Самая часто покупаемая категория (pop_cat)
    hd = data.groupby(['user_id', 'department'])['quantity'].sum().reset_index()\
                        .rename(columns={'quantity': 'cnt'})
    data = data.merge(hd, on=['user_id', 'department'], how='left')

    data = data.merge(data.groupby('user_id')['cnt'].max(), on='user_id', how='left', 
               suffixes=(None, '_max'))

    data['pop_cat'] = np.nan
    data.loc[data['cnt_max'] == data['cnt'], 'pop_cat'] = data.loc[data['cnt_max'] == data['cnt'], 'department']
    data['pop_cat'] = data.groupby('user_id')['pop_cat'].fillna(method='ffill')
    
    # Кол-во покупок юзером конкретной категории в неделю ('us_cat_quan_pw')
    hd = data.groupby(['user_id', 'department', 'week_no'])['quantity'].sum()\
                .reset_index().rename(columns={'quantity': 'us_cat_quan_pw'})
    data = data.merge(hd, on=['user_id', 'department', 'week_no'], how='left')
    
    # Средняя сумма покупки 1 товара в каждой категории (берем категорию item_id)) - (Цена item_id)
    hd = data.groupby(['user_id', 'department'])['value'].mean().reset_index()\
                        .rename(columns={'value': 'value_pcat'})
    data = data.merge(hd, on=['user_id', 'department'], how='left')

    data = data.merge(user_features, on='user_id', how='left')
    
    # Доля покупок в каждой категории среди разных возрастных групп (age_cat_q)
    hd = (data.groupby(['age_desc', 'department'])['quantity'].sum() / \
        data.groupby('age_desc')['quantity'].sum()).reset_index()\
                        .rename(columns={'quantity': 'age_cat_q'})
    data = data.merge(hd, on=['age_desc', 'department'], how='left')
    
    # Доля покупок в каждой категории среди разных групп дохода (income_cat_q)
    hd = (data.groupby(['income_desc', 'department'])['quantity'].sum() / \
        data.groupby('income_desc')['quantity'].sum()).reset_index()\
                        .rename(columns={'quantity': 'income_cat_q'})
    data = data.merge(hd, on=['income_desc', 'department'], how='left')

    # Доля покупок в каждой категории среди разных групп, имеющих детей или нет (kid_cat_q)
    hd = (data.groupby(['kid_category_desc', 'department'])['quantity'].sum() / \
        data.groupby('kid_category_desc')['quantity'].sum()).reset_index()\
                        .rename(columns={'quantity': 'kid_cat_q'})
    data = data.merge(hd, on=['kid_category_desc', 'department'], how='left')
    
    # Средняя сумма покупки 1 товара в каждой категории среди разных возрастных груп (age_val_pcat)
    hd = data.groupby(['age_desc', 'department'])['value'].mean().reset_index()\
                        .rename(columns={'value': 'age_val_pcat'})
    data = data.merge(hd, on=['age_desc', 'department'], how='left')
    
    # Средняя сумма покупки 1 товара в каждой категории среди разных групп дохода (income_val_pcat)
    hd = data.groupby(['income_desc', 'department'])['value'].mean().reset_index()\
                        .rename(columns={'value': 'income_val_pcat'})
    data = data.merge(hd, on=['income_desc', 'department'], how='left')
    
    # Средняя сумма покупки 1 товара в каждой категории среди разных групп, имеющих детей или нет (kid_val_pcat)
    hd = data.groupby(['kid_category_desc', 'department'])['value'].mean().reset_index()\
                        .rename(columns={'value': 'kid_val_pcat'})
    data = data.merge(hd, on=['kid_category_desc', 'department'], how='left')
    
    data.drop(['cnt', 'cnt_max'], axis=1, inplace=True)
    
    return data