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

def predict_cur(C_matrix, R_matrix, w_matrix, user_ratings_predict_cur):# returns all predicted values for given rating matrix based on CUR decomposition
	user_ratings_predict = np.matmul(np.matmul(C_matrix, w_matrix), R_matrix)
	return user_ratings_predict

'''
returns w matrix in CUR decomposion of given matrix(A=C(w)R), index_column = indices of selected columns of C in original matrix
index_row = indices of selected columns of R in original matrix 
'''
def findw(user_ratings,index_column,index_row,C_matrix,R_matrix,c):
	W_matrix= zeros(shape = (c, c))
	for i in range(c): 
		for j in range(c):
			W_matrix[i][j]=user_ratings[int(index_row[i][0])][int(index_column[0][j])]
	sigma= finds(W_matrix,c) #find svd decomposition for W_matrix
	print("sigma")
	print(sigma)
	u= findu(W_matrix,c)
	print("u")
	print(u)
	v= findv(W_matrix,c)
	print("v")
	print(u)
	v = v.transpose()
	u = u.transpose()
	sigma = np.linalg.pinv(sigma)# find pseudoinverse of sigma
	W_matrix = np.matmul(np.matmul(v, sigma), u) #w'' = V(sigma+)(UT)
	return W_matrix

'''
returns C matrix in CUR decomposion of given matrix(A=C(w)R), c= number of columns to be selected from given matrix
also returns indices of selected columns in original matrix
'''
def findc(user_ratings_norm,num_users,num_items,total_sum_square,c):
	column_prob = zeros(shape = (2, num_items))# column distribution for randomly selecting c columns to form C
	C_matrix= zeros(shape = (num_users, c))
	index= zeros(shape = (1, c))# matrix to store indices of selected columns in original matrix
	for i in range(num_items): 
		column_prob[0][i] = i
		for j in range(num_users):
			column_prob[1][i] = column_prob[1][i] + user_ratings_norm[j][i]*user_ratings_norm[j][i]
		column_prob[1][i] = column_prob[1][i]/total_sum_square
	j2= np.random.choice(column_prob[0], c, p=column_prob[1])# randomly selecting c indices based on column probability distribution 
	print(j2)
	for i in range(c):# randomly selecting c columns to form C based on randomly selected c indices
		j= int(j2[i])
		index[0][i]=j
		C_matrix[:,i]=(user_ratings_norm[:, j])/sqrt(c*column_prob[1][j])
	return C_matrix, index

'''
returns R matrix in CUR decomposion of given matrix(A=C(w)R), c= number of rowss to be selected from given matrix
also returns indices of selected rows in original matrix
'''
def findr(user_ratings_norm,num_users,num_items,total_sum_square,c):
	row_prob = zeros(shape = (num_users, 2))# row probability distribution for randomly selecting c rows to form R
	R_matrix= zeros(shape = (c, num_items))
	index= zeros(shape = (c, 1))# matrix to store indices of selected rows in original matrix
	for i in range(num_users): 
		row_prob[i][0] = i
		for j in range(num_items):
			row_prob[i][1] = row_prob[i][1] + user_ratings_norm[i][j]*user_ratings_norm[i][j]
		row_prob[i][1] = row_prob[i][1]/total_sum_square
	j2= np.random.choice(row_prob[:,0], c, p=row_prob[:, 1])# randomly selecting c indices based on row probability distribution 
	print(j2)
	for i in range(c):# randomly selecting c rows to form R based on randomly selected c indices
		j=int(j2[i])
		index[i][0]=j
		R_matrix[i, :]=user_ratings_norm[j , :]/sqrt(c*row_prob[j][1])
	return R_matrix, index


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

total_sum_square = 0 #variable that is equl to sum of squares of all ratings in ratings matrix
for i in range(len(item_did_rate)):
    for j in range(len(item_did_rate[i])):
        if (item_did_rate[i][j] == 1):
            total_sum_square += item_ratings[i][j]*item_ratings[i][j]
print("total_sum_square_ratings")
print(total_sum_square)
c=333 #small c to select number of rows and columns in C,R matrices for CUR decomposition 
C_matrix, index_column=findc(user_ratings,num_users,num_items,total_sum_square,c)#R matrix in CUR decomposion of given matrix(A=C(w)R)
R_matrix, index_row=findr(user_ratings,num_users,num_items,total_sum_square,c)#R matrix in CUR decomposion of given matrix(A=C(w)R)W_matrix=findw(user_ratings,index_column,index_row,C_matrix,R_matrix,c)
W_matrix=findw(user_ratings,index_column,index_row,C_matrix,R_matrix,c)#w matrix in CUR decomposion of given matrix(A=C(w)R)
user_ratings_predict_cur = zeros(shape =(num_users, num_items))# matrix that will contain predicted values for given rating matrix based on CUR decomposition
user_ratings_predict_cur= predict_cur(C_matrix, R_matrix, W_matrix, user_ratings_predict_cur)
print("C_matrix")
print(C_matrix)
print("R_matrix")
print(R_matrix)
print("W_matrix")
print(W_matrix)
print("index_column")
print(index_column)
print("index_row")
print(index_row)
print("user_ratings_predict_cur")
print(user_ratings_predict_cur)
print ("rms_user_ratings_predict_cur")
print(rms(user_ratings,num_users,num_items,user_ratings_predict_cur))
print ("Spearman_user_ratings_predict_cur")
print(Spearman(user_ratings,num_users,num_items,user_ratings_predict_cur))
relevance=2 #for calculating top_k_precision
k=3#for calculating top_k_precision
print ("top_k_precision_user_ratings_predict_cur")
print(top_k_precision(user_ratings,num_users,num_items,user_ratings_predict_cur,k,relevance))
elapsed = timeit.default_timer() - start_time
print("Time Elapsed:")
print(elapsed)