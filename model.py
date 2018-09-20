import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

'''数据读取与预处理'''
df_train = pd.read_csv('./new_data/train_set.csv')
df_test = pd.read_csv('./new_data/test_set.csv')
df_train.drop(columns=['article', 'id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

'''特征工程'''
# CountVectorizer每种词汇在该训练文本中出现的频率
vectorizer = CountVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, max_features=100000)
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])
y_train = df_train['class']-1

'''训练一个分类器'''
lg = LogisticRegression(C=4, dual=True)
lg.fit(x_train, y_train)

'''对测试集进行预测'''
y_test = lg.predict(x_test)

'''将测试的结果保存至本地'''
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('./new_data/result.csv', index=False)
print('Completed............')