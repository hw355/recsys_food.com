#!/usr/bin/env python
# coding: utf-8

# # A Recommendation Engine for The Recipes by Using Collaborative Filtering in Python

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances


# In[2]:


I = pd.read_csv('./Data_Source/interactions_train.csv')
R = pd.read_csv('./Data_Source/RAW_recipes.csv')
I.info()
R.info()


# In[3]:


I.rating.value_counts().plot(kind = 'bar', fontsize = 14,
                             figsize = (5,2)).set_title('Distribution of Rating',
                                                        fontsize = 16, ha = 'center', va = 'bottom')

plt.show()


# ## Clean up the data

# In[4]:


_all = I.drop(['date', 'u', 'i'], axis = 1)
_all


# In[5]:


grouped_1 = _all.groupby(['user_id'], as_index = False, sort = False).agg({'recipe_id':'count'}).reset_index(drop = True)
grouped_1 = grouped_1.rename(columns = {'recipe_id':'reviews_count'})
grouped_1 = grouped_1.sort_values('reviews_count', ascending = False).iloc[:7500,:]
grouped_1


# In[6]:


grouped_2 = _all.groupby(['recipe_id'], as_index = False, sort = False).agg({'user_id':'count'}).reset_index(drop = True)
grouped_2 = grouped_2.rename(columns = {'user_id':'reviews_count'})
grouped_2 = grouped_2.sort_values('reviews_count', ascending = False).iloc[:7500,:]
grouped_2


# In[7]:


_part = pd.merge(_all.merge(grouped_1).drop(['reviews_count'], axis = 1), grouped_2).drop(['reviews_count'], axis = 1)
_part


# In[8]:


print('unique users:',len(_part.user_id.unique()))
print('unique recipes:',len(_part.recipe_id.unique()))


# In[9]:


grouped_user = _part.groupby(['user_id'], as_index = False, sort = False).agg({'recipe_id':'count'}).reset_index(drop = True)
grouped_user = grouped_user.rename(columns = {'recipe_id':'reviews_count'})

grouped_recipe = _part.groupby(['recipe_id'], as_index = False, sort = False).agg({'user_id':'count'}).reset_index(drop = True)
grouped_recipe = grouped_recipe.rename(columns = {'user_id':'reviews_count'})

display(grouped_user[['reviews_count']].describe())
display(grouped_recipe[['reviews_count']].describe())


# In[10]:


_part.rating.value_counts().plot(kind = 'bar', fontsize = 14, 
                                            figsize = (5,2)).set_title('Distribution of Rating',
                                                                      fontsize = 16, ha = 'center', va = 'bottom')

plt.show()


# In[11]:


new_userID = dict(zip(list(_part['user_id'].unique()),
                      list(range(len(_part['user_id'].unique())))))
display(new_userID)

new_recipeID = dict(zip(list(_part['recipe_id'].unique()),
                      list(range(len(_part['recipe_id'].unique())))))
display(new_recipeID)


# In[12]:


df = _part.replace({'user_id': new_userID, 'recipe_id': new_recipeID})
df


# In[13]:


print('The recipes without names: ', R['id'][R['name'].isnull()].values[0])
display(df[df['recipe_id'] == R['id'][R['name'].isnull()].values[0]])


# In[16]:


recipe = R[['name', 'id', 'ingredients']].merge(_part[['recipe_id']], 
                                                left_on = 'id', right_on = 'recipe_id', 
                                                how = 'right').drop(['id'], axis = 1).drop_duplicates().reset_index(drop = True)
recipe


# In[14]:


print('unique users:',len(_part.user_id.unique()))
print('unique recipes:',len(_part.recipe_id.unique()))


# In[15]:


mean = df.groupby(['user_id'], as_index = False, sort = False).mean().rename(columns = {'rating':'rating_mean'})
df = df.merge(mean[['user_id','rating_mean']], how = 'left')
df.insert(2, 'rating_adjusted', df['rating'] - df['rating_mean'])
df


# In[17]:


train_data, test_data = train_test_split(df, test_size = 0.25)
display(train_data)
display(test_data)


# In[18]:


n_users = df.user_id.unique()
n_items = df.recipe_id.unique()

train_data_matrix = np.zeros((n_users.shape[0], n_items.shape[0]))
for row in train_data.itertuples():
    train_data_matrix[row[1]-1, row[2]-1] = row[3]

display(train_data_matrix.shape)
display(train_data_matrix)


# In[19]:


test_data_matrix = np.zeros((n_users.shape[0], n_items.shape[0]))
for row in test_data.itertuples():
    test_data_matrix[row[1]-1, row[2]-1] = row[3]

display(test_data_matrix.shape)
display(test_data_matrix)


# ## Centered cosine similarity

# In[20]:


user_similarity = 1 - pairwise_distances(train_data_matrix, metric = 'cosine')

display(user_similarity.shape)
display(user_similarity)


# In[21]:


item_similarity = 1 - pairwise_distances(train_data_matrix.T, metric = 'cosine')

display(item_similarity.shape)
display(item_similarity)


# ## Predict the ratings

# In[22]:


def predict(ratings, similarity, _type = 'user'):
    if _type == 'user':
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis = np.newaxis)])
    
    elif _type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis = 1)]) 
    
    return pred


# In[23]:


user_pred = predict(train_data_matrix, user_similarity, _type = 'user')

display(user_pred.shape)
display(user_pred)


# In[24]:


user_pred_df = pd.DataFrame(user_pred, columns = list(n_items))
user_pred_df.insert(0, 'user_id', list(n_users))

user_pred_df


# In[25]:


item_pred = predict(train_data_matrix, item_similarity, _type = 'item')

display(item_pred.shape)
display(item_pred)


# In[26]:


item_pred_df = pd.DataFrame(item_pred, columns = list(n_items))
item_pred_df.insert(0, 'user_id', list(n_users))

item_pred_df


# ## Evaluation of the predictions

# In[27]:


def RMSE(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    
    return sqrt(mean_squared_error(prediction, ground_truth))


# In[28]:


user_RMSE = RMSE(user_pred, test_data_matrix)
item_RMSE = RMSE(item_pred, test_data_matrix)

print('user_RMSE = {}'.format(user_RMSE))
print('item_RMSE = {}'.format(item_RMSE))


# ## The recommendation engine

# In[29]:


def getRecommendations_UserBased(user_id, top_n = 10):
    for old_user, new_user in new_userID.items():
        if user_id == new_user:
            print(f'Top {top_n} Recommended Recipes for Original User ID: {old_user}\n')
    
    movie_rated = list(df['recipe_id'].loc[df['user_id'] == user_id])
    _all = user_pred_df.loc[user_pred_df['user_id'] == user_id].copy()
    _all.drop(user_pred_df[movie_rated], axis = 1, inplace = True)
    unwatch_sorted = _all.iloc[:,1:].sort_values(by = _all.index[0], axis = 1, ascending = False)
    dict_top_n = unwatch_sorted.iloc[:, :top_n].to_dict(orient = 'records')

    i = 1
    for recipe_id in list(dict_top_n[0].keys()):
        for old_recipe, new_recipe in new_recipeID.items():
            if recipe_id == new_recipe:
                name = recipe[recipe['recipe_id'] == old_recipe]['name'].values[0]
                ingredients = recipe[recipe['recipe_id'] == old_recipe]['ingredients'].values[0]

                print(f'Top {i} Original Recipe ID: {old_recipe} - {name}\n Ingredients: {ingredients}\n')
                
                i += 1
                
    return dict_top_n[0]


# In[31]:


R1_UserBased = getRecommendations_UserBased(702)
R1_UserBased


# In[32]:


R1_ItemBased = getRecommendations_ItemBased(702)
R1_ItemBased


# In[33]:


R2_UserBased = getRecommendations_UserBased(408, 5)
R2_UserBased


# In[34]:


R2_ItemBased = getRecommendations_ItemBased(408, 5)
R2_ItemBased


# In[35]:


R3_UserBased = getRecommendations_UserBased(204, 7)
R3_UserBased


# In[36]:


R3_ItemBased = getRecommendations_ItemBased(204, 7)
R3_ItemBased

