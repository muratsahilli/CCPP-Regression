import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
#######################################################################################################
# read data

df = pd.read_csv('C:/Users/murat/Downloads/CCPP/CCPP/Folds5x2_pp.txt')
df.head()

col_name = ['sıcaklık','egzoz vakum', 'ortam basıncı', 'bağıl nem',
            'elektrik enerjisi']
df.columns = col_name
df.head()

col_name_inputs = ['sıcaklık','egzoz vakum', 'ortam basıncı', 'bağıl nem']
data_scaled = StandardScaler().fit_transform(df)
dfscale = pd.DataFrame(data_scaled)
dfscale.columns = col_name

sns.heatmap(np.abs(df.corr()), annot = True)
######################## Compute each feature Linear Regression and MSE  ###############################
lr_mse_array = []
lr_mse_array_format = []

for i in col_name_inputs:  
    x1 = dfscale[i].values.reshape(-1,1)
    y1 = dfscale['elektrik enerjisi']
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(x1, y1, test_size=0.2, random_state=0)
    
    lr = LinearRegression()
    lr.fit(X_train_1, y_train_1)
    
    y_train_pred_1 = lr.predict(X_train_1)
    y_test_pred_1 = lr.predict(X_test_1)
    # Mean Squared Error (MSE)
    mse_train_1 = mean_squared_error(y_train_1, y_train_pred_1)
    mse_test_1 = mean_squared_error(y_test_1, y_test_pred_1)
    lr_mse_array.append(mse_test_1)
    
    # Virgülden sonra 2 basamak alarak başka bir array'e atıyoruz
    lr_mse_array_format.append("{:.2f}".format(mse_test_1))
    
# Barplot
mse_array_series = pd.DataFrame(lr_mse_array)
# Plot the figure.
plt.figure(figsize=(12, 8))
ax = mse_array_series.plot(kind='bar')
ax.set_title('MSE Barplot with Linear Regression')
ax.set_ylabel('Mse values')
ax.set_xticklabels(col_name_inputs,rotation=0)
rects = ax.patches


for rect, label in zip(rects, lr_mse_array_format):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height- 0.05, label,
            ha='center', va='bottom',color='white')

#----------------------------------------------------------------------------

# make new dataframe with the half of the columns of dataframe which MSE values are lowest 
half_data_mse = []
list_for_colnames_mse = [] 

for i in range(len(lr_mse_array)//2):
    #take highest value's  index number
    imin = np.argmin(lr_mse_array)
    #get data from original dataframe with index number
    min_values= dfscale[col_name[imin]].to_numpy()
    half_data_mse.append(min_values)
    #change highest value to lowest
    lr_mse_array[imin] += (1 + lr_mse_array[np.argmax(lr_mse_array)])
    list_for_colnames_mse.append(col_name[imin])

   
half_data_mse.append(dfscale['elektrik enerjisi'].to_numpy())
list_for_colnames_mse.append('elektrik enerjisi')
df_half_data_mse =  pd.DataFrame(half_data_mse).T
df_half_data_mse.columns = list_for_colnames_mse


# veri seti içindeki değişkenlerin dağılımlarının çizdirilmesi
df_half_data_mse.hist(bins=10,figsize=(16,9),grid=False);
# veri seti içindeki değişkenlerin ilişki katsayılarının çizdirilmesi
print("Veri seti içindeki değişkenlerin birbiri ile ilişki katsayısı")
corr=np.abs(df_half_data_mse.corr())
plt.figure()

sns.heatmap(corr, annot = True)

features= df_half_data_mse.drop(columns=['elektrik enerjisi'])
elektrik = df_half_data_mse['elektrik enerjisi']
# Verileri çıkış değerine karşılık çizdirme
plt.figure(figsize=(20, 5))
for i, col in enumerate(features.columns):
    # 3 plots here hence 1, 3
    plt.subplot(1, len(features.columns), i+1)
    x = df_half_data_mse[col]
    y = elektrik
    plt.plot(x, y, 'o')
    # Create regression line
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('elektrik enerjisi')

X = df_half_data_mse.iloc[:, :-1].values
Y = df_half_data_mse['elektrik enerjisi'].values
seed = 7
num_folds = 10
RMS = 'neg_mean_squared_error'
validation_size = 0.20

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)
#denenecek modellerin yüklenmesi
models = []
models.append(('LR', LinearRegression()))
models.append(('RIDGE', Ridge()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('RF', RandomForestRegressor()))
models.append(('SVR', SVR()))

# algoritmalara k-fold uyguladığımızda çıkan 10 mse ve r2 değerlerinin ortalamasını tutmak için
lr_list = []
ridge_list = []
lasso_list = []
en_list = []
knn_list = []
dt_list = []
rf_list = []
svr_list = []

r2_results=[]
mse_results = []
names = []

for name, model in models:
    
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = np.abs(cross_val_score(model, X_train, Y_train, cv=kfold, scoring=RMS))
    r2_scores =  cross_val_score(model, X_train, Y_train, cv=kfold, scoring='r2')
    
    mse_results.append(cv_results)    
    r2_results.append(r2_scores) 
    names.append(name)
    
    # algorithms k-fold mse r2 comparison table(dataframe)
    if name =='LR':
        lr_list.append(cv_results.mean())
        lr_list.append(r2_scores.mean())
    if name =='RIDGE':
        ridge_list.append(cv_results.mean())
        ridge_list.append(r2_scores.mean())
    if name =='LASSO':
        lasso_list.append(cv_results.mean())
        lasso_list.append(r2_scores.mean())
    if name =='EN':
        en_list.append(cv_results.mean())
        en_list.append(r2_scores.mean())
    if name =='KNN':
        knn_list.append(cv_results.mean())
        knn_list.append(r2_scores.mean())
    if name =='DT':
        dt_list.append(cv_results.mean())
        dt_list.append(r2_scores.mean())
    if name =='RF':
        rf_list.append(cv_results.mean())
        rf_list.append(r2_scores.mean())
    if name =='SVR':
        svr_list.append(cv_results.mean())
        svr_list.append(r2_scores.mean())
    
    msg = "%s mse: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    r2sc = "%s r2: %f (%f)" %(name, r2_scores.mean(), r2_scores.std())
    print(" {}\n {}".format(msg,r2sc))
    
    kfolddf = pd.DataFrame(kfold.split(X_train))

    index_max = np.argmax(r2_scores)
    train_index, test_index = kfolddf.iloc[index_max]
    X_train_kf, X_test_kf, y_train_kf, y_test_kf = X_train[train_index], X_train[test_index], Y_train[train_index], Y_train[test_index]
    model.fit(X_train_kf, y_train_kf)
    curveaxis=np.zeros((100,X_test_kf.shape[1]))
    for cx in range(X_test_kf.shape[1]):
        curveaxis[:,cx]=np.linspace(np.min(X_test_kf[:,cx]),np.max(X_test_kf[:,cx]),100) # linspace komutu başlangıç ve bitiş değerleri arasında belirtilen sayı kadar(100) parçalı değer oluşturur 
    curve_predictions = model.predict(curveaxis) 
    
    #tahmin ve rezidü çizimleri
    train_predictions = model.predict(X_train_kf)
    test_predictions = model.predict(X_test_kf)
    print(name," Test Prediction MSE",":", mean_squared_error(y_test_kf, test_predictions))
    print(name,"Train Prediction MSE",":", mean_squared_error(y_train_kf, train_predictions))    
    
    #residual plot
    plt.figure(figsize=(12,8))
    plt.scatter(train_predictions, train_predictions - y_train_kf, c='blue', marker='o', label='Training data')
    plt.scatter(test_predictions, test_predictions - y_test_kf, c='orange', marker='*', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.title(name+' Residual Analysis')
    plt.show()
    
    #plot curve
    plt.title(name + ' Curve') # çizim başlığı
    plt.scatter(X_test_kf[:,0], y_test_kf,c='b') # test verisi 
    plt.scatter(X_test_kf[:,0], test_predictions,c='r',alpha=0.5) # test verisine karşılık prediction
    plt.plot(curveaxis[:,0], curve_predictions,c='black')# 0 sütunu değer atamaya karşılık tahminlerin eğri olarak çizilmesi
    plt.show()
    
    label = model
    #mse ve r2 
    folds=['KF1', 'KF2', 'KF3','KF4','KF5','KF6','KF7','KF8','KF9','KF10']
    r2_plotdf = pd.DataFrame({'r2 scores':r2_scores}, index=folds)
    ax = r2_plotdf.plot.bar(rot=0)
    ax.set_ylabel(label)    
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),rotation=90,color='red')
    ax.legend(loc="lower center")
    plt.show()
    
    mse_plotdf = pd.DataFrame({'mse scores':cv_results}, index=folds)
    ax = mse_plotdf.plot.bar(rot=0)
    ax.set_ylabel(label)  
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),rotation=90,color='red')
    ax.legend(loc="lower center")
    plt.show()
    
# k-fold mse ve r2 ortalamaları tablosu için dataframe
d = {'K-Fold-MSE-PrePro': ['mse-mean', 'r2-mean'], 'LR': lr_list, 'RIDGE': ridge_list, 'LASSO': lasso_list,
     'EN': en_list, 'KNN': knn_list, 'DT': dt_list, 'RF': rf_list, 'SVR': svr_list}
df2 = pd.DataFrame(data=d)

# r2 ve mse mean değerlerini yazdırma , en yüksek doğruluk
print("\n")  
r2_mean=[]
for i in range(len(names)):
    print(names[i],r2_results[i].mean())
    r2_mean.append(r2_results[i].mean())
print("\n En yüksek doğruluk(Accuracy) değeri:")
print(names[r2_mean.index(max(r2_mean))], max(r2_mean))
print("\n")
mse_mean=[]
for i in range(len(names)):
    print(names[i],mse_results[i].mean())
    mse_mean.append(mse_results[i].mean())
print("\n En yüksek doğruluk(Accuracy) değeri:")
print(names[mse_mean.index(min(mse_mean))], min(mse_mean))

# r2 mean bar plot
labels=['LR','RIDGE','LASSO','EN','KNN','DT','RF','SVR']
r2_mean_plotdf = pd.DataFrame({'r2 scores COMPARISON':r2_mean}, index=labels)
ax = r2_mean_plotdf.plot.bar(rot=0)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),rotation=90,color='red')
ax.legend(loc="lower right")
plt.show()

# mse mean bar plot
mse_mean_plotdf = pd.DataFrame({'mse scores COMPARISON':mse_mean}, index=labels)
ax = mse_mean_plotdf.plot.bar(rot=0)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),rotation=90,color='red')
ax.legend(loc="upper right")
plt.show()