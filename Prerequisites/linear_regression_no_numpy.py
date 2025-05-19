
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

def dot(a,b):

    return sum(a[i]*b[i] for i in range(len(a)))


def cost(X, y, w, b):

    m = len(X)

    total_cost = 0

    for i in range (m):
        
        f = dot(X[i],w)+b

        total_cost += (y[i]-f)**2/2
    
    total_cost /= m

    return total_cost


def gradients(X, y, w, b):

    m, n = len(X), len(X[0])
    dj_dw = [0 for _ in range(n)]
    dj_db = 0.

    for i in range(m):
        
        f = dot(X[i],w)+b
        
        dj_db += (f-y[i])
        
        for j in range(n):
            dj_dw[j] += (f-y[i])*X[i][j]
            
    dj_dw = [x/m for x in dj_dw]
    dj_db = dj_db/m

        
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

        w_in = [w_in[j] - alpha * dj_dw[j] for j in range(len(w_in))]
                   
        b_in = b_in - alpha * dj_db              
        
        if i<100000:      
            cost =  cost_function(X, y, w_in, b_in)
            J_history.append(cost)

        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float((2*J_history[-1])**0.5):8.2f}   ")
        
    return w_in, b_in, J_history, w_history 


def predict(X,w_in,b_in):

    pred=[]

    for i in X:
        prediction=dot(w_in,i)+b_in
        pred.append(prediction)

    return pred


def r2_score(y_true, y_pred):
    
    y_true = list(y_true)
    y_pred = list(y_pred)
    
    mean_true = sum(y_true) / len(y_true)
    ss_res = sum((yt-yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt-mean_true) ** 2 for yt in y_true)

    return 1.0 -ss_res/ss_tot 