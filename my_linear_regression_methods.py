
# coding: utf-8

# In[6]:


import numpy as np


# In[9]:


def mserror(y, y_pred):
    y_pred = y_pred.squeeze()
    ans = sum((y_pred - y) ** 2) / len(y)
    return ans


# In[2]:


def normal_equation(X, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)


# In[3]:


def linear_prediction(X, w):
    return np.dot(X, w)


# In[4]:


def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    grad = (2. / len(y)) * X[train_ind, :] * (np.dot(X[train_ind,:], w) - y[train_ind])
    return  w - eta * grad[:,np.newaxis]


# In[1]:


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом. 
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Счетчик итераций
    iter_num = 0
    # Будем порождать псевдослучайные числа 
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)
        
    # Основной цикл
    while weight_dist > min_weight_dist and iter_num < max_iter:
        # порождаем псевдослучайный 
        # индекс объекта обучающей выборки
        random_ind = np.random.randint(X.shape[0])

        w_old = w
        w = stochastic_gradient_step(X, y, w, random_ind, eta=eta)
        weight_dist = np.linalg.norm(w - w_old)
        errors.append(mserror(y, np.dot(X, w)))
        
        if iter_num % 10000 == 0 and verbose:
            print("Iteration: ", iter_num)
            
        iter_num += 1    
        
    return w, errors

