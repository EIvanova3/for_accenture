
# coding: utf-8

# In[314]:


import numpy as np
import pandas as pd
import statsmodels.stats.api as sm
import seaborn as sns
from xgboost import XGBRegressor
import my_linear_regression_methods as my_lr
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, ARDRegression
from sklearn.linear_model import Lasso, Ridge, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
get_ipython().run_line_magic('pylab', 'inline')


# In[315]:


data = pd.read_csv('task2.txt', sep='\t', header=None)
data.head()


# Проверим на наличие пустых элементов

# In[316]:


data.isnull().values.any()


# Описание данных, чтобы понять нужна ли предобработка

# In[317]:


data.describe()


# Признаки имеют одинаковый масштаб, значит для дальнейшей работы нам лучше ненормировать матрицу объекты-признаки.

# In[318]:


# Для масштабирования признаков, выше обоснован отказ от этого, т.к. масштаб у них один.
# from sklearn.preprocessing import scale
# from sklearn.utils import shuffle
# df_shuffled = shuffle(data, random_state=123)
# X_s = scale(df_shuffled[df_shuffled.columns[:-1]])
# y_s = df_shuffled[y_index]
# pd.DataFrame(X_s).describe()


# In[319]:


# масштабирование признаков незначительно, но увеличило ошибку, работаю дальше с не масштабированными признаками
# means, stds = np.mean(X, axis = 0), np.std(X, axis = 0)
# X_scale = (X - means) / stds
# X_scale_with_ones = np.hstack((X_scale, np.ones((len(X_scale), 1), dtype=float))) # добавляю единичный признак
# print("Cреднеквадратичная ошибка прогноза, если всегда предсказывать медианное значение отклика по исходной выборке: %f\n"  % my_lr.mserror(y, np.median(y) * np.ones(len(y))))
# norm_eq_weights_scale = my_lr.normal_equation(X_scale_with_ones, y)
# print("веса для масштабированных признаков:", norm_eq_weights_scale)
# error_norm_scale = my_lr.mserror(y, my_lr.linear_prediction(X_scale_with_ones, norm_eq_weights_scale))
# print("ср. кв. ошибка по вектору весов масштабированных данных", error_norm_scale)


# In[320]:


y_index = data.shape[1] - 1


# Посчитаем корреляции всех признаков, кроме последнего, с последним с помощью метода corrwith:

# In[321]:


data.loc[:,data.columns[:-1]].corrwith(data[y_index])


# In[322]:


data.loc[:,:].corr()


# На диагоналях, как и полагается, стоят единицы. И кореллирующих между собой признаков нет.

# Построим гистограмму распределения каждого признака из выборки data. 

# In[323]:


# [data.plot(y=[x], kind='hist', color='red', title='1') for x in range(y_index)]


# Отобразим попарные зависимости признаков. По диагонали рисуются гистограммы распределения признаков, а вне диагонали - scatter plots зависимости двух признаков.

# In[324]:


sns.pairplot(data)


# Ярко выраженных выбросов в данных нет, как и зависимостей признаков междусобой, что отмечается и корреляцией.
# Начиная с 5 признака зависимость похожа на константную, в дальнейшем буду обучать и сравнивать модели как на всех признаках, так и на первых 5 только.

# In[325]:


X, y = data.values[:, :-1], data.values[:, -1]


# Посмотрим на распределение ответов

# In[326]:


# Counter(data[y_index])
# data.groupby(y_index)[y_index].count()


# In[327]:


plt.figure(figsize(8,6))
stat = data.groupby(y_index)[y_index].agg(lambda x : float(len(x)) / data.shape[0])
stat.plot(kind='bar', fontsize=12, width=0.9, color="red")
plt.xticks(rotation=0)
plt.ylabel('Proportion', fontsize=14)
plt.xlabel(str(y_index) + ' column', fontsize=14)


# Распределение ответов похоже на нормальное, поэтому буду решать задачу регресии 

# # 1 Линейная регрессия и стохастический градиентный спуск

# Добавляю единичный признак

# In[328]:


X_with_ones = np.hstack((X, np.ones((len(X), 1), dtype=float)))


# Функция mserror - среднеквадратичная ошибка прогноза. Она принимает два аргумента - объекты Series y (значения целевого признака) и y_pred (предсказанные значения).
# 
# Cреднеквадратичная ошибка прогноза, если всегда предсказывать медианное значение отклика по исходной выборке:

# In[329]:


mean_squared_error(y, np.median(y) * np.ones(len(y)))


# Функция normal_equation, которая по заданным матрицам (массивам NumPy) X и y вычисляет вектор весов w согласно нормальному уравнению линейной регрессии

# In[330]:


norm_eq_weights = my_lr.normal_equation(X_with_ones, y)
print('Веса:',  norm_eq_weights)
error_norm = mean_squared_error(y, my_lr.linear_prediction(X_with_ones, norm_eq_weights))
print('Среднеквадратичная ошибка:', error_norm)


# In[331]:


get_ipython().run_cell_magic('time', '', 'stoch_grad_desc_weights, stoch_errors_by_iter = my_lr.stochastic_gradient_descent(X_with_ones, y, np.zeros((X_with_ones.shape[1],1)), eta=1e-2, max_iter=1e5, min_weight_dist=1e-8, seed=0, verbose=True)')


# Посмотрим, чему равна ошибка на первых 50 итерациях стохастического градиентного спуск.

# In[332]:


plt.plot(range(50), list(stoch_errors_by_iter[:50]))
xlabel('Iteration number')
ylabel('MSE')


# Теперь посмотрим на зависимость ошибки от номера итерации для $10^5$ итераций стохастического градиентного спуска. Видим, что алгоритм сходится.

# In[333]:


plot(range(len(stoch_errors_by_iter)), stoch_errors_by_iter)
xlabel('Iteration number')
ylabel('MSE')


# Посмотрим на вектор весов, к которому сошелся метод и сравним его с вектором весов из аналитического решения

# In[334]:


stoch_grad_desc_weights.squeeze(), norm_eq_weights


# In[335]:


mean_squared_error(y, np.dot(X_with_ones, stoch_grad_desc_weights)), error_norm


# Видим, что среднеквадратичная ошибка при весах полученных стохастическим градиентным спуском больше, чем при аналитическом решении. И такое бывает! Скорее всего зависимость нелинейная, но поанализируем еще немножко, чтобы было с чем сравнивать! 

# # 2

# In[336]:


linear_regressor = LinearRegression()
linear_regressor.fit(X,y)
print ("признак -> коэффициент\n")
for pair in zip(data.columns, linear_regressor.coef_):
    print (pair)
print("\nкоэффициент детерминации:", linear_regressor.score(X, y))


# In[337]:


# Обучим линейную модель с L1-регуляризацией и выведим веса
lasso_regressor = Lasso()
lasso_regressor.fit(X,y)
print ("признак -> коэффициент\n")
for pair in zip(data.columns, lasso_regressor.coef_):
    print (pair)
print("\nкоэффициент детерминации:", lasso_regressor.score(X, y))


# Весы Lasso нулевые при очень большом alpha. Чем больше alpha, тем ниже сложность модели, т.е. меньше признаков участвуют в построении. Самая простая модель константа, т.е. ни один признак не участвует -> все веса равны нулю. alpha = 1. При alpha = 0 получаем веса как в модели линейной регрессии.

# In[338]:


# Обучим линейную модель с L1-регуляризацией и выведим веса
lasso_regressor = Lasso(alpha=0.)
lasso_regressor.fit(X,y)
print ("признак -> коэффициент\n")
for pair in zip(data.columns, lasso_regressor.coef_):
    print (pair)
print("\nкоэффициент детерминации:", lasso_regressor.score(X, y))


# Значит нам нужно выбрать наилучшее alpha на отрезке [0, 1]. 
# 
# Будем использовать в качестве метрики сам оптимизируемый функционал метода наименьших квадратов, то есть Mean Square Error.
# И делать несколько разбиений выборки, на каждом пробовать разные значения alpha, а затем усреднять MSE. Удобнее всего делать такие разбиения кросс-валидацией. 

# In[339]:


alphas = np.arange(0., 1., 0.0005)
lasso_regressor_CV = LassoCV(alphas=alphas, cv=3)
lasso_regressor_CV.fit(X,y)

plt.figure(figsize=(8, 5))
plt.plot(lasso_regressor_CV.alphas_, lasso_regressor_CV.mse_path_.mean(axis = 1))
plt.xlabel("alpha")
plt.ylabel("averaged MSE")
plt.title("LassoCV")
print ("alpha = %f\n"  % (lasso_regressor_CV.alpha_))
print ("признак -> коэффициент\n")
for pr, w in zip(data.columns[:-1], lasso_regressor_CV.coef_):
    print ("'%str' -> %f" % (pr, w))


# In[340]:


# Выведим значения alpha, соответствующие минимумам MSE на каждом разбиении (то есть по столбцам).
print ("alphas_min -> mse_argmin")
for a, mse_min in zip(lasso_regressor_CV.alphas_[np.argmin(lasso_regressor_CV.mse_path_, axis = 0)],lasso_regressor_CV.mse_path_.min(axis = 0)):
    print("%f -> %f" % (a, mse_min))

plt.rcParams['figure.figsize'] = 20, 5

iter = 1
while (iter<=3):
    plt.subplot(1,3,iter)
    plt.plot(lasso_regressor_CV.alphas_, lasso_regressor_CV.mse_path_[:,iter-1])
    plt.title(iter)
    plt.xlabel('alpha')
    plt.ylabel('MSE')
    iter+=1


# In[341]:


lasso_regressor = Lasso(0.014)
lasso_regressor.fit(X,y)
for pair in zip(data.columns, lasso_regressor.coef_):
    print (pair)
lasso_regressor.score(X, y)


# In[342]:


# Обучим линейную модель с L2-регуляризацией и выведим веса
ridge_regressor = Ridge()
ridge_regressor.fit(X,y)
for pair in zip(data.columns, ridge_regressor.coef_):
    print (pair)
ridge_regressor.score(X, y)


# In[343]:


ridge_regressor = Ridge(alpha=0.014)
ridge_regressor.fit(X,y)
for pair in zip(data.columns, ridge_regressor.coef_):
    print (pair)
ridge_regressor.score(X, y)


# # 3

# Научимся оценивать ответы. Отделим 25% выборки для контроля качества предсказания:

# In[344]:


np.random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns[:-1]], data[data.columns[-1]], test_size=0.3)
print("среднее значение отклика обучающей выборки: %f" % np.mean(y_train))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error([np.mean(y_train)]*len(y_train), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error([np.mean(y_train)]*len(y_test), y_test)))


# ## Линейная регрессия

# In[345]:


lm = LinearRegression()
lm.fit(X_train, y_train)
np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(lm.predict(X_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(lm.predict(X_train), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(lm.predict(X_test), y_test)))
print('коэффициент детерминации: %f' % lm.score(X_test, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, lm.predict(X_test)))


# In[346]:


print (list(np.array(y_test)[:10]))
print (list(map(lambda x: int(round(x)), (lm.predict(X_test))[:10])))


# ## SGD регрессия

# In[347]:


sgd = SGDRegressor(random_state = 0, max_iter = 100)
sgd.fit(X_train, y_train)
np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(sgd.predict(X_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(sgd.predict(X_train), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(sgd.predict(X_test), y_test)))
print('коэффициент детерминации: %f' % sgd.score(X_test, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, sgd.predict(X_test)))


# In[348]:


sgd.get_params().keys()


# In[349]:


parameters_grid = {
    'max_iter' : [3, 10, 50, 100], 
    'penalty' : ['l1', 'l2', 'none'],
    'alpha' : [0., 0.01, 0.014, 0.1],
}


# In[350]:


grid_cv = GridSearchCV(sgd, parameters_grid, scoring = 'mean_absolute_error', cv = 4)


# In[351]:


get_ipython().run_cell_magic('time', '', 'grid_cv.fit(X_train, y_train)')


# In[352]:


print (grid_cv.best_score_)
print (grid_cv.best_params_)


# In[353]:


np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(grid_cv.best_estimator_.predict(X_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(grid_cv.best_estimator_.predict(X_train), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(grid_cv.best_estimator_.predict(X_test), y_test)))
print('коэффициент детерминации: %f' % grid_cv.best_estimator_.score(X_test, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, grid_cv.best_estimator_.predict(X_test)))


# In[354]:


print (list(np.array(y_test)[:10]))
print (list(map(lambda x: int(round(x)), (grid_cv.best_estimator_.predict(X_test))[:10])))


# In[355]:


pylab.figure(figsize=(16, 6))

pylab.subplot(1,2,1)
pylab.grid(True)
pylab.scatter(y_train, sgd.predict(X_train), alpha=0.5, color = 'red')
pylab.scatter(y_test, sgd.predict(X_test), alpha=0.5, color = 'blue')
pylab.title('no parameters setting')
pylab.xlim(-10,50)
pylab.ylim(-10,50)

pylab.subplot(1,2,2)
pylab.grid(True)
pylab.scatter(y_train, grid_cv.best_estimator_.predict(X_train), alpha=0.5, color = 'red')
pylab.scatter(y_test, grid_cv.best_estimator_.predict(X_test), alpha=0.5, color = 'blue')
pylab.title('grid search')
pylab.xlim(-10,50)
pylab.ylim(-10,50)


# ## ARD регрессия

# In[356]:


ard = ARDRegression()
ard.fit(X_train, y_train)
np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(ard.predict(X_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(ard.predict(X_train), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(ard.predict(X_test), y_test)))
print('коэффициент детерминации: %f' % ard.score(X_test, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, ard.predict(X_test)))


# In[357]:


print (list(np.array(y_test)[:10]))
print (list(map(lambda x: int(round(x)), (ard.predict(X_test))[:10])))


# ## Случайный лес

# Построим на обучающей выборке случайный лес:

# In[358]:


rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(X_train, y_train)
np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(rf.predict(X_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(rf.predict(X_train), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(rf.predict(X_test), y_test)))
print('коэффициент детерминации: %f' % rf.score(X_test, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, rf.predict(X_test)))


# In[359]:


print (list(np.array(y_test)[:10]))
print (list(map(lambda x: int(round(x)), (rf.predict(X_test))[:10])))


# In[360]:


pylab.figure(figsize=(16, 6))

pylab.subplot(1,2,1)
pylab.grid(True)
pylab.scatter(y_train, lm.predict(X_train), alpha=0.5, color = 'red')
pylab.scatter(y_test, lm.predict(X_test), alpha=0.5, color = 'blue')
pylab.title('LinearRegression')
pylab.xlim(-10,50)
pylab.ylim(-10,50)

pylab.subplot(1,2,2)
pylab.grid(True)
pylab.scatter(y_train, rf.predict(X_train), alpha=0.5, color = 'red')
pylab.scatter(y_test, rf.predict(X_test), alpha=0.5, color = 'blue')
pylab.title('RandomForestRegressor')
pylab.xlim(-10,50)
pylab.ylim(-10,50)


# Сравним ошибки линейной регрессии и случайного леса на тестовой выборке:

# In[361]:


plt.figure(figsize(8,6))
plt.hist(abs(y_test - lm.predict(X_test)) - abs(y_test - rf.predict(X_test)), bins=15, normed=True)
plt.xlabel('Difference of absolute errors')


# Различия между средними абсолютными ошибками значимы:

# In[362]:


tmeans = sm.CompareMeans(sm.DescrStatsW(abs(y_test - lm.predict(X_test))), 
                         sm.DescrStatsW(abs(y_test - rf.predict(X_test))))
print('Средняя разность абсолютных ошибок: %f' % np.mean(abs(y_test - lm.predict(X_test)) - abs(y_test - rf.predict(X_test))))
tmeans.ttest_ind(alternative='two-sided', usevar='pooled', value=0)[1]


# 95% доверительный интервал для средней разности абсолютных ошибок:

# In[363]:


tmeans.tconfint_diff(alpha=0.05, alternative='two-sided', usevar='pooled')


# Посмотрим, какие признаки обладают наибольшей предсказательной способностью:

# In[364]:


importances = pd.DataFrame([[k, l] for k, l in zip(X_train.columns, rf.feature_importances_)])
importances.columns = ['feature name', 'importance']
importances.sort_values(by='importance', ascending=False)


# Cильнее всего на ответ влияет признак 3, записанный в 4 ом столбце

# # 4

# Наименее значимые 5, 6, 7, 8 признаки попробуем исключить 5, 7, 8 так как их зануляет и l1 регуляризатор в линейной регрессии

# In[365]:


X_train_new, X_test_new =  X_train.loc[:, [0, 1, 2, 3, 4, 6, 9]], X_test.loc[:, [0, 1, 2, 3, 4, 6, 9]]


# In[366]:


lm_h = LinearRegression()
lm_h.fit(X_train_new, y_train)
np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(lm_h.predict(X_train_new)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(lm_h.predict(X_train_new), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(lm_h.predict(X_test_new), y_test)))
print('коэффициент детерминации: %f' % lm_h.score(X_test_new, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, lm_h.predict(X_test_new)))


# In[367]:


sgd_h = SGDRegressor(random_state = 0, max_iter = 100)
sgd_h.fit(X_train_new, y_train)
np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(sgd_h.predict(X_train_new)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(sgd_h.predict(X_train_new), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(sgd_h.predict(X_test_new), y_test)))
print('коэффициент детерминации: %f' % sgd_h.score(X_test_new, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, sgd_h.predict(X_test_new)))


# In[368]:


ard_h = ARDRegression()
ard_h.fit(X_train_new, y_train)
np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(ard_h.predict(X_train_new)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(ard_h.predict(X_train_new), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(ard_h.predict(X_test_new), y_test)))
print('коэффициент детерминации: %f' % ard_h.score(X_test_new, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, ard_h.predict(X_test_new)))


# In[369]:


rf_h = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf_h.fit(X_train_new, y_train)
np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(rf_h.predict(X_train_new)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(rf_h.predict(X_train_new), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(rf_h.predict(X_test_new), y_test)))
print('коэффициент детерминации: %f' % rf_h.score(X_test_new, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, rf_h.predict(X_test_new)))


# # 5

# случайный лес для сравнения
# 
# среднее значение отклика обучающей выборки: 15.049629
# 
# корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: 0.909693
# 
# корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: 2.060640
# 
# коэффициент детерминации: 0.829803
# 
# абсолютная ошибка: 1.579129

# In[370]:


gb = GradientBoostingRegressor(n_estimators=100, min_samples_leaf=3)
gb.fit(X_train, y_train)
np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(gb.predict(X_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(gb.predict(X_train), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(gb.predict(X_test), y_test)))
print('коэффициент детерминации: %f' % gb.score(X_test, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, gb.predict(X_test)))


# In[414]:


gb_importances = pd.DataFrame([[k, l] for k, l in zip(X_train.columns, gb.feature_importances_)])
gb_importances.columns = ['feature name', 'importance']
gb_importances.sort_values(by='importance', ascending=False)


# In[371]:


xgb = XGBRegressor()
xgb.fit(X_train, y_train)
np.random.seed(0)
print("среднее значение отклика обучающей выборки: %f" % np.mean(xgb.predict(X_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, обучение: %f" % sqrt(mean_squared_error(xgb.predict(X_train), y_train)))
print("корень из среднеквадратичной ошибки прогноза средним значением на обучающей выборке, тест: %f" % sqrt(mean_squared_error(xgb.predict(X_test), y_test)))
print('коэффициент детерминации: %f' % xgb.score(X_test, y_test))
print('абсолютная ошибка: %f' % mean_absolute_error(y_test, xgb.predict(X_test)))


# In[372]:


print (list(np.array(y_test)[50:60]))
print (list(map(lambda x: int(round(x)), (xgb.predict(X_test))[50:60])))


# In[376]:


xgb_importances = pd.DataFrame([[k, l] for k, l in zip(X_train.columns, xgb.feature_importances_)])
xgb_importances.columns = ['feature name', 'importance']
xgb_importances.sort_values(by='importance', ascending=False)


# In[373]:


pylab.figure(figsize=(16, 6))

pylab.subplot(1,3,1)
pylab.grid(True)
pylab.scatter(y_train, lm.predict(X_train), alpha=0.5, color = 'red')
pylab.scatter(y_test, lm.predict(X_test), alpha=0.5, color = 'blue')
pylab.title('LinearRegression')
pylab.xlim(0,30)
pylab.ylim(0,30)

pylab.subplot(1,3,2)
pylab.grid(True)
pylab.scatter(y_train, rf.predict(X_train), alpha=0.5, color = 'red')
pylab.scatter(y_test, rf.predict(X_test), alpha=0.5, color = 'blue')
pylab.title('RandomForesRegressor')
pylab.xlim(0,30)
pylab.ylim(0,30)

pylab.subplot(1,3,3)
pylab.grid(True)
pylab.scatter(y_train, xgb.predict(X_train), alpha=0.5, color = 'red')
pylab.scatter(y_test, xgb.predict(X_test), alpha=0.5, color = 'blue')
pylab.title('XGBRegressor')
pylab.xlim(0,30)
pylab.ylim(0,30)


# Сравним ошибки линейной регрессии и градиентного бустинга на тестовой выборке:

# In[416]:


plt.figure(figsize(8,6))
plt.hist(abs(y_test - lm.predict(X_test)) - abs(y_test - xgb.predict(X_test)), bins=15, normed=True)
plt.xlabel('Difference of absolute errors')


# Различия между средними абсолютными ошибками значимы:

# In[418]:


tmeans_lm_xgb = sm.CompareMeans(sm.DescrStatsW(abs(y_test - lm.predict(X_test))), 
                         sm.DescrStatsW(abs(y_test - xgb.predict(X_test))))
print('Средняя разность абсолютных ошибок: %f' % np.mean(abs(y_test - lm.predict(X_test)) - abs(y_test - xgb.predict(X_test))))
tmeans_lm_xgb.ttest_ind(alternative='two-sided', usevar='pooled', value=0)[1]


# 95% доверительный интервал для средней разности абсолютных ошибок:

# In[419]:


tmeans_lm_xgb.tconfint_diff(alpha=0.05, alternative='two-sided', usevar='pooled')


# Сравним ошибки случайного леса и градиентного бустинга на тестовой выборке:

# In[421]:


plt.figure(figsize(8,6))
plt.hist(abs(y_test - rf.predict(X_test)) - abs(y_test - xgb.predict(X_test)), bins=15, normed=True)
plt.xlabel('Difference of absolute errors')


# Различия между средними абсолютными ошибками значимы:

# In[422]:


tmeans_rf_xgb = sm.CompareMeans(sm.DescrStatsW(abs(y_test - rf.predict(X_test))), 
                         sm.DescrStatsW(abs(y_test - xgb.predict(X_test))))
print('Средняя разность абсолютных ошибок: %f' % np.mean(abs(y_test - rf.predict(X_test)) - abs(y_test - xgb.predict(X_test))))
tmeans_rf_xgb.ttest_ind(alternative='two-sided', usevar='pooled', value=0)[1]


# 95% доверительный интервал для средней разности абсолютных ошибок:

# In[423]:


tmeans_rf_xgb.tconfint_diff(alpha=0.05, alternative='two-sided', usevar='pooled')


# **Как по коэффициенту детерминации, так и по среднеквадратичной ошибке однозначно лучшей моделью является градиентный бустинг, в частности, его реализация в пакете xgboost.**
# 
# **Наибольшей предсказательной способностью согласно моделям линейной регрессии и случайного леса обладает 3 признак, если нумерация начинается с 0.**
# 
# **Согласно лучшей по метрикам качества модели градиентного бустинга наибольшей предсказательной способностью обладает признак под номером 0, для реализации модели из пакета xgboost.**
# 
# **В то время как для модели градиентного бустинга из пакета sklearn наибольшей предсказательной способностью обладает признак под номером 2.** 
# 
