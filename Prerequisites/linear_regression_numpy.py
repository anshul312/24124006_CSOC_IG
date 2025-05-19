import numpy as np
import random,math,time


def train_test_split(data,test_size=0.25,random_state=42):

    random.seed(random_state)

    total_size=len(data)
    num_size=int(test_size*total_size)

    indices=list(range(total_size))
    random.shuffle(indices)

    test_indices=indices[:num_size]
    train_indices=indices[num_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


def cost(X, y, w, b):

    m = len(X)
    preds = np.dot(X, w) + b               
    errors = y - preds                     
    total_cost = (errors ** 2).sum() / (2 * m)
    
    return total_cost

def gradients(X, y, w, b):

    m, n = X.shape
    dj_db = 0.

    pred=np.dot(X,w)+b  
    error=(pred-y)     
    dj_db = error.sum()/m     
    dj_dw = np.dot(X.T,error ) / m            
    
    return dj_db, dj_dw


def timer(original_func):

    def wrapper(*args,**kwargs):
        init_time=time.perf_counter()
        result=original_func(*args,**kwargs)
        final_time=time.perf_counter()

        print('{} ran in:{} sec'.format(original_func.__name__,final_time-init_time))

        return result

    return wrapper


@timer
def gradient_descent(X, y, w_in, b_in, cost_function, gradients, alpha, num_iters): 
        
    J_history = []
    w_history = []

    for i in range(num_iters):

        dj_db, dj_dw = gradients(X, y, w_in, b_in)   

        w_in = w_in - alpha * dj_dw 
                   
        b_in = b_in - alpha * dj_db              
        
        if i<100000:      
            cost =  cost_function(X, y, w_in, b_in)
            J_history.append(cost)

        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float((2*J_history[-1])**0.5):8.2f}   ")
        
    return w_in, b_in, J_history, w_history 


def predict(X,w_in,b_in):

    return np.dot(X, w_in) + b_in 


def r2_score(y_true, y_pred):
    
    y_true = list(y_true)
    y_pred = list(y_pred)
    
    mean_true = sum(y_true) / len(y_true)
    ss_res = sum((yt-yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt-mean_true) ** 2 for yt in y_true)

    return 1.0 -ss_res/ss_tot 