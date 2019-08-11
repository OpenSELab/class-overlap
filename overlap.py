import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import metrics




if __name__ == '__main__':
    df = pd.read_csv('D:\\myoverlap\\Data\\NASA\\PC4.csv')
    x = df.iloc[:, :df.shape[1]-1].values
    y = df.iloc[:, df.shape[1]-1].values
    print(df.shape)
    x = np.log1p(x)

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.15)

    # scaler = MinMaxScaler()
    # x=scaler.fit_transform(x)



    k=int(df.shape[0]*0.85/20)

    random_state = 170

    kmean= KMeans(n_clusters=k, random_state=random_state)
    y_pred=kmean.fit_predict(x_train)

    center=kmean.cluster_centers_
   # print(center)


    df_x=pd.DataFrame(x_train)
    df_y=pd.DataFrame(y_train)
    df_y_pred=pd.DataFrame(y_pred)
    df_new=pd.concat((df_x,df_y,df_y_pred),axis=1)
    df_new.columns = ['LOC_BLANK','BRANCH_COUNT','CALL_PAIRS','LOC_CODE_AND_COMMENT','LOC_COMMENTS','CONDITION_COUNT',
                        'CYCLOMATIC_COMPLEXITY','CYCLOMATIC_DENSITY','DECISION_COUNT','DECISION_DENSITY','DESIGN_COMPLEXITY',
                        'DESIGN_DENSITY','EDGE_COUNT','ESSENTIAL_COMPLEXITY','ESSENTIAL_DENSITY','LOC_EXECUTABLE',
                        'PARAMETER_COUNT','HALSTEAD_CONTENT','HALSTEAD_DIFFICULTY','HALSTEAD_EFFORT','HALSTEAD_ERROR_EST',
                        'HALSTEAD_LENGTH','HALSTEAD_LEVEL','HALSTEAD_PROG_TIME','HALSTEAD_VOLUME','MAINTENANCE_SEVERITY',
                        'MODIFIED_CONDITION_COUNT','MULTIPLE_CONDITION_COUNT','NODE_COUNT','NORMALIZED_CYLOMATIC_COMPLEXITY',
                        'NUM_OPERANDS','NUM_OPERATORS','NUM_UNIQUE_OPERANDS','NUM_UNIQUE_OPERATORS','NUMBER_OF_LINES',
                        'PERCENT_COMMENTS','LOC_TOTAL','Defective','label']
    print(df_new.shape)

    df_Numpy=df_new.iloc[:, df_new.shape[1]-1].values
    #print(df_Numpy)
    df_x=df_new.iloc[:, :df.shape[1]-1].values
    #print(df_Numpy.shape)
    over=np.zeros((df_new.shape[0],1))

    for i in range(df_new.shape[0]):
        l=int(df_Numpy[i])
        op=np.linalg.norm(df_x[i]-center[l])
        for j in range(k):
           if j!=l:
               oq=np.linalg.norm(df_x[i]-center[j])
               ol=min(op/(op+oq),oq/(op+oq))
               over[i]=over[i]+ol
        over[i]=over[i]/(k-1)

    #print(over)

    df_over=pd.DataFrame(over)
    #print(df_over.shape)

    df_new = pd.concat((df_new,df_over), axis=1)
    df_new.columns = ['LOC_BLANK', 'BRANCH_COUNT', 'CALL_PAIRS', 'LOC_CODE_AND_COMMENT', 'LOC_COMMENTS',
                      'CONDITION_COUNT',
                      'CYCLOMATIC_COMPLEXITY', 'CYCLOMATIC_DENSITY', 'DECISION_COUNT', 'DECISION_DENSITY',
                      'DESIGN_COMPLEXITY',
                      'DESIGN_DENSITY', 'EDGE_COUNT', 'ESSENTIAL_COMPLEXITY', 'ESSENTIAL_DENSITY', 'LOC_EXECUTABLE',
                      'PARAMETER_COUNT', 'HALSTEAD_CONTENT', 'HALSTEAD_DIFFICULTY', 'HALSTEAD_EFFORT',
                      'HALSTEAD_ERROR_EST',
                      'HALSTEAD_LENGTH', 'HALSTEAD_LEVEL', 'HALSTEAD_PROG_TIME', 'HALSTEAD_VOLUME',
                      'MAINTENANCE_SEVERITY',
                      'MODIFIED_CONDITION_COUNT', 'MULTIPLE_CONDITION_COUNT', 'NODE_COUNT',
                      'NORMALIZED_CYLOMATIC_COMPLEXITY',
                      'NUM_OPERANDS', 'NUM_OPERATORS', 'NUM_UNIQUE_OPERANDS', 'NUM_UNIQUE_OPERATORS', 'NUMBER_OF_LINES',
                      'PERCENT_COMMENTS', 'LOC_TOTAL', 'Defective','label', 'over']

    df_new.sort_values('over', inplace=True,ascending=False)
    #print(df_new['over'])

    p=int(0.8*df_new.shape[0])

    x_train_new= df_new.iloc[:p, :df.shape[1]-1].values
    y_train_new=df_new.iloc[:p, df.shape[1]-1].values

    s=SVC()

    s.fit(x_train,y_train)

    pred=s.predict(x_test)

    accu = metrics.accuracy_score(y_test, pred)
    auc = metrics.roc_auc_score(y_test, pred)
    mc = metrics.matthews_corrcoef(y_test, pred)
    print(metrics.classification_report(y_test, pred))
    print(metrics.confusion_matrix(y_test, pred))
    print(auc, mc)

    s.fit(x_train_new, y_train_new)

    pred1 = s.predict(x_test)

    accu1 = metrics.accuracy_score(y_test, pred1)
    auc1 = metrics.roc_auc_score(y_test, pred1)
    mc1 = metrics.matthews_corrcoef(y_test, pred1)
    print(metrics.classification_report(y_test, pred1))
    print(metrics.confusion_matrix(y_test, pred1))
    print(auc1, mc1)

  #   center=kmean.cluster_centers_
  #
  #   print(center)
  #   plt.figure(figsize=(2, 12))
  #   plt.subplot(221)  # 在2图里添加子图2
  #   plt.scatter(x[:, 0], x[:, 1],x[:, 2], c=y)
  #
  #   plt.subplot(222)  # 在2图里添加子图2
  #   plt.scatter(x[:, 0], x[:, 1],x[:, 2], c=y_pred)
  # #  plt.title("Anisotropicly Distributed Blobs")
  #
  #   plt.show()
  #