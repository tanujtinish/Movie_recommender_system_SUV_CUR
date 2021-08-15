import numpy as np
from numpy import *
import math
import timeit

'''
Function to calculate the rms error on two matrices which are predicted and actual
'''
def rms(user_ratings,user_did_rate,num_users,num_items,user_ratings_predict):
    tot = 0
    ans = 0
    for i in range(int(3*num_users/4),num_users):
        for j in range(int(3*num_items/4),num_items):
            if(user_did_rate[i][j] == 1 and user_ratings_predict[i][j] !=1):
                ans += (user_ratings[i][j] - user_ratings_predict[i][j])*(user_ratings[i][j] - user_ratings_predict[i][j])
                tot += 1
    ans /= tot
    ans = math.sqrt(abs(ans))
    return ans

'''
Function to calculate the Spearman error on two matrices which are predicted and actual
'''
def Spearman(user_ratings,num_users,num_items,user_ratings_predict):
    tot = 0
    ans = 0
    for i in range(int(3*num_users/4),num_users):
        for j in range(int(3*num_items/4),num_items):
            if(user_ratings_predict[i][j] !=1):
                ans += (user_ratings[i][j] - user_ratings_predict[i][j])*(user_ratings[i][j] - user_ratings_predict[i][j])
                tot += 1
    ans /= tot
    ans = math.sqrt(abs(ans))
    ans=1-(ans*6)/(tot*tot-1)
    return ans

'''
Function to calculate Top K precision error on two matrices which are predicted and actual
'''

def top_k_precision(user_ratings,num_users,num_items,user_ratings_predict,k,relevance):
    total=0
    x2=np.ndarray.tolist(user_ratings)
    x=np.ndarray.tolist(user_ratings_predict.astype(int))
    for i in range(num_users):
        x3=[w for q, w in sorted(zip(x[i], x[i]))]
        x4=[w for q, w in sorted(zip(x[i], x2[i]))]
        actual_items=0
        predicted_items=0
        for j in range(k):
            if (int(x3[num_items-j-1])>=relevance):
                actual_items=actual_items+1
                if(x4[num_items-j-1]>=relevance):
                    predicted_items=predicted_items+1
        if (actual_items>0):
            total=total+(predicted_items/actual_items)
    total=total/num_users
    return (total)


'''
Function to Normalize movie ratings by substracting mean from non-zero elements
'''
def normalize_item_ratings(item_ratings, item_did_rate, num_items):
    item_ratings_mean = zeros(shape = (num_items, 2))
    item_ratings_norm = zeros(shape = item_ratings.shape)
    for i in range(num_items): 
        idx = where(item_did_rate[i] == 1)[0]
        item_ratings_mean[i][1] = mean(item_ratings[i, idx])
        item_ratings_mean[i][0] = i
        item_ratings_norm[i, idx] = item_ratings[i, idx] - item_ratings_mean[i][1]            
    return item_ratings_norm, item_ratings_mean


'''
Function to Normalize user ratings by substracting mean from non-zero elements
'''
def normalize_user_ratings(user_ratings, user_did_rate, num_users,num_items):
    user_ratings_mean = zeros(shape = (num_users, 2))
    user_ratings_norm = zeros(shape = user_ratings.shape)
    
    for i in range(num_users): 
        
        idx = where(user_did_rate[i] == 1)[0]
        
        user_ratings_mean[i][1] = mean(user_ratings[i, idx])
        user_ratings_mean[i][0] = i
        user_ratings_norm[i, idx] = user_ratings[i, idx] - user_ratings_mean[i][1]
    return user_ratings_norm, user_ratings_mean


'''
Function to calculate cosine similarity between two vectors
'''
def cosine_similarity(a,b):
    x = np.dot(a,a)
    y = np.dot(b,b)
    if(x == 0 or y == 1):
        x = 1
        y = 1
    return np.dot(a,b)/math.sqrt(x*y)

'''
Function which returns a matrix of dimension given by (noOfUsers x noOfUsers) where each element a[i][j] is cosine similarity between i and j users
'''
def similarity_user(user_ratings_norm, users):
    user_similarity_cosine = np.ones((users,users))
    for user1 in range(num_users):
        for user2 in range(num_users):
            if(user1 > user2):
                user_similarity_cosine[user1][user2] = cosine_similarity(user_ratings_norm[user1],user_ratings_norm[user2])
                user_similarity_cosine[user2][user1] = user_similarity_cosine[user1][user2]
    return user_similarity_cosine

'''
Function which calculates the top most similar users to a given user using cosine similarity
'''
def most_similar_users(user_ratings_norm,user1,k, user_did_rate,item,num_users,user_similarity_cosine):
    scores = np.empty(shape = [0, 2])
    for user2 in range(num_users):
        if  (user2 != user1 and user_did_rate[user2][item] == 1):
            scores = np.append(scores, [[user_similarity_cosine[user1][user2],user2]], axis=0)
    scores = scores[np.argsort(scores[:, 0])]
    total=0
    scores = scores[len(scores)-k:len(scores)]
    scores = np.array(scores)
    for y in range(len(scores)):
        total=total+ scores[y][0]
    return scores, total

'''
Function to predict the unknown ratings and it returns the predicted matrix using baseline approach
'''
def predict_collab_baseline(user_similarity_cosine, user_ratings_norm, k, user_ratings_predict, user_ratings,user_did_rate,num_users, item_ratings_mean, user_ratings_mean, average_total,user_ratings_predict1):
    for i in range(int(3*len(user_ratings_predict)/4),len(user_ratings_predict)):
        for j in range(int(3*len(user_ratings_predict[0])/4),len(user_ratings_predict[0])):
            scores, total= most_similar_users(user_ratings_norm,i,k,user_did_rate,j,num_users,user_similarity_cosine)
            ans=0
            for y in range(len(scores)):
                ans= ans + scores[y][0]*(user_ratings[int(scores[y][1])][j]-((item_ratings_mean[j][1] + user_ratings_mean[int(scores[y][1])][1] - average_total)))
            if(total == 0):
                total = 1
            ans= ans/total
            ans= ans + item_ratings_mean[j][1] + user_ratings_mean[i][1] - average_total
            user_ratings_predict[i][j] = ans
    return user_ratings_predict

#Input No of users, movies and ratings 
start_time = timeit.default_timer()
with open('u.info', 'r') as f:
    for line in f:
        y = line.split()
        if(y[1]=="users"):
            num_users = int(y[0])
        elif(y[1]=="items"):
            num_items = int(y[0])
        elif(y[1]=="ratings"):
            num_ratings = int(y[0])
item_ratings = zeros(shape = (num_items, num_users))
#Input Ratings into matrix
with open('u.data', 'r') as f:
    for line in f:
        y = line.split()
        item_ratings[int(y[1])-1][int(y[0])-1]=int(y[2])
item_did_rate = (item_ratings != 0) * 1
print ("item_ratings")
print (item_ratings)
item_ratings_norm, item_ratings_mean = normalize_item_ratings(item_ratings, item_did_rate, num_items)       
#Input Ratings into matrix
user_ratings = zeros(shape = (num_users,num_items))
with open('u.data', 'r') as f:
    for line in f:
        y = line.split()
        user_ratings[int(y[0])-1][int(y[1])-1]=int(y[2])
user_did_rate = (user_ratings != 0) * 1
print ("user_ratings")
print (user_ratings)
average_total = 0
for i in range(len(item_did_rate)):
    for j in range(len(item_did_rate[i])):
        if (item_did_rate[i][j] == 1):
            average_total += item_ratings[i][j]
average_total /= num_ratings

user_ratings_norm, user_ratings_mean = normalize_user_ratings(user_ratings, user_did_rate, num_users,num_items)
user_similarity_cosine= similarity_user(user_ratings, num_users)

user_ratings_predict_collab_baseline = zeros(shape = (num_users, num_items))
#Set K to a value and print the required values and also errors
k=40
user_ratings_predict_collab_baseline = predict_collab_baseline(user_similarity_cosine, user_ratings_norm, k, user_ratings_predict_collab_baseline, user_ratings,user_did_rate,num_users, item_ratings_mean, user_ratings_mean, average_total,user_ratings_predict_collab_baseline)
print ("user_ratings_predict_collab_baseline")
print (user_ratings_predict_collab_baseline)
print ("rms_user_ratings_predict_collab_baseline")
print(rms(user_ratings,user_did_rate,num_users,num_items,user_ratings_predict_collab_baseline))
print ("spearman_user_ratings_predict_collab_baseline")
print(Spearman(user_ratings,num_users,num_items,user_ratings_predict_collab_baseline))
relevance=2 #for calculating top_k_precision
k=3 #for calculating top_k_precision
print ("top_k_precision_user_ratings_predict_collab_baseline")
print(top_k_precision(user_ratings,num_users,num_items,user_ratings_predict_collab_baseline,k,relevance))
elapsed = timeit.default_timer() - start_time
print("Time Elapsed:")
print(elapsed)
