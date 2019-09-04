
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.multiclass import unique_labels

def shifting(name):
    # read companies data
    df = pd.read_csv('stock_adjclosed.csv', index_col=0)
    name_list = df.columns.values.tolist()
    df.fillna(0,inplace=True)
    
    for i in range(1,6):
        df[name+'_'+str(i)] = (df[name].shift(-i) -df[name])/df[name]
    
    df.fillna(0,inplace=True)
    return name_list, df


def extract_x_y(name):
    name_list, df = shifting(name)
    
    label = []
    threshold = 0.02
    for i in range(df.shape[0]):
        if df[name+'_1'][i]>threshold or df[name+'_2'][i]>threshold or df[name+'_3'][i]>threshold or df[name+'_4'][i]>threshold or df[name+'_5'][i]>threshold :
            label.append(1)
        elif df[name+'_1'][i]< -threshold or df[name+'_2'][i]< -threshold or df[name+'_3'][i]< -threshold or df[name+'_4'][i]< -threshold or df[name+'_5'][i] < -threshold:
            label.append(-1)
        else:
            label.append(0)
    
    df['label'] = label             
    
    df.fillna(0, inplace= True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    df_pct_change = df[[name for name in name_list]].pct_change()
    df_pct_change = df_pct_change.replace([np.inf, -np.inf], 0)
    df_pct_change.fillna(0, inplace=True)
    
    str_labels = []
    for i in label:
        str_labels.append(str(i)) 
    print(Counter(str_labels))
    
    X = df_pct_change
    y = df['label']
    return X, y

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def split_train_test(name):
    X,y = extract_x_y(name)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)
    return X_train, X_test, y_train, y_test

def feature_scaling(X_train, X_test):
    from sklearn import preprocessing
    minmaxscaler = preprocessing.StandardScaler()
    X_train = pd.DataFrame(minmaxscaler.fit_transform(X_train))
    X_test = pd.DataFrame(minmaxscaler.transform(X_test))

    return X_train, X_test

def filter_method(X_train,X_test):  
    
    #Removing Correlated Features
    correlated_features = set()  
    correlation_matrix = X_train.corr() 
    
    for i in range(len(correlation_matrix .columns)):  
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
                
    print(correlated_features)  
    
    
    # Drop Correlated data
    X_train.drop(labels=correlated_features, axis=1, inplace=True)  
    X_test.drop(labels=correlated_features, axis=1, inplace=True)
     
    
    return X_train, X_test

def do_ml(name, X_train, X_test, y_train, y_test):
    
    classifer = GradientBoostingClassifier(n_estimators=200, random_state = 0)    
#    clf = KNeighborsClassifier()
    
    classifer.fit(X_train, y_train)
    y_pred = classifer.predict(X_test)
    print('Predicted spread:', Counter(y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy',accuracy)
    class_names = np.array(['hold','buy','sell'])

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=class_names,title='Confusion matrix')
    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')


def main():
   name = 'MMM'
   name_list, df = shifting(name)
   X_train, X_test, y_train, y_test = split_train_test(name)
   X_train, X_test = feature_scaling(X_train, X_test)
   X_train, X_test = filter_method(X_train,X_test)
   do_ml(name, X_train, X_test, y_train, y_test)
if __name__ == '__main__':
   main()
    
    

