import numpy as np
from numpy import *
import math
import timeit

def rms(user_ratings,num_users,num_items,user_ratings_predict): #takes actual matrix and predicted marix for error calculation
    tot = 0 # variable to store total number of ratings
    ans = 0 # variable to store final rms error
    for i in range(int(3*num_users/4),num_users): #loop through last 25percent rows
        for j in range(int(3*num_items/4),num_items): #loop through last 25percent columns
            if(user_ratings_predict[i][j] !=1): #to consider only those values which have been rated
                ans += (user_ratings[i][j] - user_ratings_predict[i][j])*(user_ratings[i][j] - user_ratings_predict[i][j])
                tot += 1
    ans /= tot
    ans = math.sqrt(abs(ans))
    return ans

def Spearman(user_ratings,num_users,num_items,user_ratings_predict): #takes actual matrix and predicted marix for spearman error calculation
    tot = 0 # variable to store total number of ratings
    ans = 0 # variable to store rms error
    for i in range(int(3*num_users/4),num_users): #loop through last 25percent rows
        for j in range(int(3*num_items/4),num_items): #loop through last 25percent columns
            if(user_ratings_predict[i][j] !=1): #to consider only those values which have been rated
                ans += (user_ratings[i][j] - user_ratings_predict[i][j])*(user_ratings[i][j] - user_ratings_predict[i][j])
                tot += 1
    ans /= tot
    ans = math.sqrt(abs(ans)) # here ans is equal to rms
    ans=1-(ans*6)/(tot*tot-1)
    return ans

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


def gram_schmidt(matrix):# converts matrix into an orthogonal matrix by applying the Gram-Schmidt orthonormalization process to the column vectors
    newmatrix = []
    for v in matrix: #loop for all rows in matrix
    	sum=0
    	for row in newmatrix: 
    		sum+= np.matmul(v,row)*row
    	w = v - sum #w is row in new matrix correspondingto row in given matrix
    	newmatrix.append(w/np.linalg.norm(w))
    newmatrix=np.array(newmatrix) #converting newmatrix from list to matrix
    return newmatrix



def predict_svd(sigma, u ,v, user_ratings_predict):# returns all predicted values for given rating matrix based on SVD decomposition
	user_ratings_predict = np.matmul(np.matmul(u, sigma), v)
	return user_ratings_predict

def finds(user_ratings_norm,n): # returns sigma matrix in svd decomposion of given matrix(A=U(sigma)Vt), n = number of columns in matrix
	user_ratings_norm_transpose = user_ratings_norm.transpose() # find transpose of given matrix
	result= np.matmul(user_ratings_norm, user_ratings_norm_transpose) # find A*A(t)
	eigenvalue, eigenvector= np.linalg.eig(result) # find eigenvalues and their eigenvectors for matrix A*A(t)
	eigenvalue.sort() # sort eigenvalues for finding sigma matrix
	total=0
	total2=0
	for i in range(len(eigenvector[0])):
		total=total+eigenvalue[i]*eigenvalue[i]

	n1 = len(eigenvector[0])
	sigma = zeros(shape = (len(eigenvector[0]),n)) # initializing matrix sigma
	for i in range(min(len(eigenvector[0]),n)):
		sigma[i][i] = math.sqrt(abs(eigenvalue[n1-i-1])) # sigma is diagnol matrix with values as sqrt of eigenvalues in decreasing order
	return sigma

def findu(user_ratings_norm,n):# returns U matrix in svd decomposion of given matrix(A=U(sigma)Vt), n = number of columns in matrix
	user_ratings_norm_transpose = user_ratings_norm.transpose()# find transpose of given matrix(At)
	result= np.matmul(user_ratings_norm, user_ratings_norm_transpose)# find A*A(t)
	eigenvalue, eigenvector= np.linalg.eig(result)# find eigenvalues and their eigenvectors for matrix A*A(t)
	sigma= finds(user_ratings_norm,n)# find sigma matrix in svd decomposion of given matrix(A=U(sigma)Vt)
	for passnum in range(len(eigenvalue)-1):# this loop re-arrange matrix columns wrt reverse-sorting of eigenvalues such that lowest column contains eigenvector corresponding to maximum eigenvalue
		for i in range(len(eigenvalue)-passnum-1):
			if (eigenvalue[i]<eigenvalue[i+1]):
				temp = np.copy(eigenvector[:,i])
				eigenvector[:,i] = eigenvector[:,i+1]
				eigenvector[:,i+1] = temp
	v= findv(user_ratings_norm,n)#find Vt matrix in svd decomposion of given matrix(A=U(sigma)Vt)
	u=gram_schmidt(eigenvector)#Apply Gram-Schmidt orthonormalization to matrix of eigenvectors
	for i in range(len(sigma)) :# loop that selects appropriate eigenvectors depending on given matrix
		temp = np.dot(user_ratings_norm,np.matrix(v[i]).T)
		tempU = np.matrix(u[:,i]).T
		flag = False
		for j in range(len(temp)):
			if tempU[j]!= 0.0:
				if temp[j]/tempU[j]<0.0:
					flag= True
					break
		if flag :
			for k in range(len(u[:,i])):
				u[k][i]= -1*(u[k][i])
	return u

def findv(user_ratings_norm,n):#returns Vt matrix in svd decomposion of given matrix(A=U(sigma)Vt), n = number of columns in matrix
	user_ratings_norm_transpose = user_ratings_norm.transpose()# find transpose of given matrix(At)
	result= np.matmul(user_ratings_norm_transpose, user_ratings_norm)# find A*A(t)
	eigenvalue, eigenvector= np.linalg.eig(result)# find eigenvalues and their eigenvectors for matrix A*A(t)
	sigma= finds(user_ratings_norm,n)# find sigma matrix in svd decomposion of given matrix(A=U(sigma)Vt)
	for passnum in range(len(eigenvalue)-1):# this loop re-arrange matrix columns wrt reverse-sorting of eigenvalues such that lowest column contains eigenvector corresponding to maximum eigenvalue
		for i in range(len(eigenvalue)-passnum-1):
			if (eigenvalue[i]<eigenvalue[i+1]):
				temp = np.copy(eigenvector[:,i])
				eigenvector[:,i] = eigenvector[:,i+1]
				eigenvector[:,i+1] = temp
	v=gram_schmidt(eigenvector)#Apply Gram-Schmidt orthonormalization to matrix of eigenvectors to get V
	v = v.transpose()# find transpose to get Vt
	return v

start_time = timeit.default_timer()
with open('u.info', 'r') as f:# read number of users, number of items, and total number of ratings from file
    for line in f:
        y = line.split()
        if(y[1]=="users"):
        	num_users = int(y[0])
        elif(y[1]=="items"):
        	num_items = int(y[0])
        elif(y[1]=="ratings"):
        	num_ratings = int(y[0])

item_ratings = zeros(shape = (num_items, num_users))# matrix which will contain ratings of all movies by some users, each row corresponds to movie, and each column corresponds to user
with open('u.data', 'r') as f: # read data from file to make item_ratings mattrix
    for line in f:
        y = line.split()
        item_ratings[int(y[1])-1][int(y[0])-1]=int(y[2])
item_did_rate = (item_ratings != 0) * 1 #user_did_rate matrix contains 1 if movie was rated by user, otherwise 0
print("item_ratings")
print (item_ratings)		
		
user_ratings = zeros(shape = (num_users,num_items))# matrix which will contain ratings of all users for some movies, each row corresponds to user, and each column corresponds to movie
with open('u.data', 'r') as f:# read data from file to make user_ratings mattrix
    for line in f:
        y = line.split()
        user_ratings[int(y[0])-1][int(y[1])-1]=int(y[2])
user_did_rate = (user_ratings != 0) * 1 #user_did_rate matrix contains 1 if userr rated movie, otherwise 0
print("user_ratings")
print (user_ratings)

sigma= finds(user_ratings,num_items)#sigma matrix in SVD decomposion of given matrix(A=U(sigma)Vt)
print("sigma done")
u= findu(user_ratings,num_items)#U matrix in SVD decomposion of given matrix(A=U(sigma)Vt)
print("u done")
v= findv(user_ratings,num_items)#Vt matrix in SVD decomposion of given matrix(A=U(sigma)Vt)
print("v done")
print("u")
print(u)
print("v")
print(v)
print("sigma")
print(sigma)

user_ratings_predict_svd = zeros(shape =(num_users, num_items))# matrix that will contain predicted values for given rating matrix based on SVD decomposition
user_ratings_predict_svd= predict_svd(sigma, u ,v, user_ratings_predict_svd)
print("user_ratings_predict_svd")
print(user_ratings_predict_svd)
print ("rms_user_ratings_predict_svd")
print(rms(user_ratings,num_users,num_items,user_ratings_predict_svd))
print ("Speaman_user_ratings_predict_svd")
print(Spearman(user_ratings,num_users,num_items,user_ratings_predict_svd))
relevance=2 #for calculating top_k_precision
k=3#for calculating top_k_precision
print ("top_k_precision_user_ratings_predict_svd")
print(top_k_precision(user_ratings,num_users,num_items,user_ratings_predict_svd,k,relevance))
elapsed = timeit.default_timer() - start_time
print("Time Elapsed:")
print(elapsed)