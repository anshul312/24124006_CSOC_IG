import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score,average_precision_score

def relu(z):
    return np.maximum(z,0)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def relu_derivative(z):
    return (z > 0).astype(float)

def int_to_onehot(y, num_labels):
    one_hot = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        one_hot[i, val] = 1
    return one_hot

class NeuralNet:

    def __init__(self,num_classes,num_features,num_hidden,random_seed=42):

        self.num_classes=num_classes
        
        rng=np.random.RandomState(random_seed)
        std_h = np.sqrt(2. / num_features)
        self.weight_h=rng.normal(loc=0.0,scale=std_h,size=(num_hidden,num_features))
        self.bias_h = np.zeros(num_hidden)

        std_out=np.sqrt(2./num_hidden)
        self.weight_out = rng.normal(loc=0.0,scale=std_out,size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self,x):

        z_h=np.dot(x,self.weight_h.T)+self.bias_h
        a_h= relu(z_h)

        z_out = np.dot(a_h,self.weight_out.T)+self.bias_out
        a_out=softmax(z_out)
        return z_h,a_h, a_out
    

    def backward(self,x,z_h,a_h,a_out,y):

        y_onehot=int_to_onehot(y,self.num_classes)

        delta_out=(a_out-y_onehot)/y.shape[0]

        d_loss__dw_out=np.dot(delta_out.T,a_h) 
        d_loss__db_out = np.sum(delta_out, axis=0)

        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        d_a_h__d_z_h = relu_derivative(z_h)
        d_z_h__d_w_h = x
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T,
        d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)


        return (d_loss__dw_out, d_loss__db_out,d_loss__d_w_h, d_loss__d_b_h)        

def mini_batch_generator(X,y,minibatch_size,rng):
    
    indices=np.arange(y.shape[0])
    rng.shuffle(indices)

    for start_idx in range(0,indices.shape[0]-minibatch_size+1,minibatch_size):

        batch_idx= indices[start_idx:start_idx+minibatch_size]

        yield X[batch_idx],y[batch_idx]

def cross_entropy_loss(targets,probas,num_labels):

    one_hot_targets=int_to_onehot(targets,num_labels)

    log_probs = np.log(probas + 1e-15)

    loss = -np.sum(one_hot_targets * log_probs) / targets.shape[0]
    return loss

def accuracy(targets, predicted_labels):
    
    return np.mean(targets==predicted_labels)


def compute_loss_and_acc(nnet, X, y,rng, num_labels=2,minibatch_size=100,threshold=0.20):

    error, correct_pred, num_examples = 0., 0, 0
    all_targets = []
    all_preds = []

    minibatch_gen = mini_batch_generator(X, y, minibatch_size,rng)
    for i, (features, targets) in enumerate(minibatch_gen):

        _,__, probas = nnet.forward(features)
        
        if nnet.num_classes == 2:
            predicted_labels = (probas[:, 1] >= threshold).astype(int)
        else:
            predicted_labels = np.argmax(probas, axis=1)

        all_targets.extend(targets)
        all_preds.extend(predicted_labels)
        
        loss = cross_entropy_loss(targets, probas,num_labels)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        error += loss
    error = error/i
    acc = correct_pred/num_examples
    f1 = f1_score(all_targets, all_preds, average='binary' if num_labels == 2 else 'macro')
    

    return error, acc, f1

def train(model,X_train,y_train,X_valid,y_valid,num_epochs,minibatch_size,learning_rate=0.01):
    
    train_loss=[]
    validation_loss=[]
    epoch_train_acc=[]
    epoch_valid_acc = []

    rng = np.random.RandomState(42)


    for _ in range(num_epochs):

        mini_batch_gen=mini_batch_generator(X_train,y_train,minibatch_size,rng=rng)

        for X_mini,y_mini in mini_batch_gen:

            z_h,a_h,a_out=model.forward(X_mini)

            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_mini,z_h, a_h, a_out,y_mini)

            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out


        train_err, train_acc,f1_on_train = compute_loss_and_acc(
        model, X_train, y_train,rng=rng
        )
        valid_err, valid_acc,f1_on_test = compute_loss_and_acc(
        model, X_valid, y_valid,rng=rng
        )
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        train_loss.append(train_err)
        validation_loss.append(valid_err)
        if((_+1)%10==0):
            print(f'Epoch: {_+1:03d}/{num_epochs:03d} '
            f'| Train loss: {train_err:.3f} '
            f'| Train F1:{f1_on_train:.2f}'
            f'| Train Acc: {train_acc:.2f}% '
            f'| Valid Acc: {valid_acc:.2f}%'
            f'| Valid F1:{f1_on_test:.2f}')
    

    return train_loss, validation_loss,epoch_train_acc, epoch_valid_acc


def final_metrics(model,X_valid,y_valid,threshold):

    _,__, probas = model.forward(X_valid)
    predicted_labels = (probas[:, 1] >= threshold).astype(int)

    f1=f1_score(y_valid,predicted_labels)
    precision=precision_score(y_valid,predicted_labels)
    recall=recall_score(y_valid,predicted_labels)
    
    PR_AUC=average_precision_score(y_valid,probas[:,1])

    return f1,precision,recall,predicted_labels,PR_AUC








