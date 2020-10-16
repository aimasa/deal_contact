from pro_data import process_data
from contact_classify import normal_param
import numpy as np
def load_corpus():
    process_data_init = process_data.process_data()
    process_data_init.split_data_file(normal_param.train_path)
    # process_data_init.deal_data(part= 0, n_part= process_data_init.all_text_path)
    x_texts_labels = process_data_init.build_datas_and_labels(process_data_init.all_text_path)
    # 是取文本中对应的词语在该文本中的对应下标
    x_texts = [x.split(" ") for x, label in x_texts_labels]
    # list_arr_text = self.deal_text(x_texts)
    labels = [label for x, label in x_texts_labels]
    return x_texts, labels

from sklearn.preprocessing import OneHotEncoder


corpus, labels = load_corpus()




from gensim import models, corpora

dic = corpora.Dictionary(corpus)
clean_data = [dic.doc2bow(words) for words in corpus]


tf_idf=models.TfidfModel(clean_data)
corpus_tfidf=tf_idf[clean_data]
# 查看主题
lda = models.ldamodel.LdaModel(clean_data, id2word=dic,  num_topics=20)

d_l=[]
labelss=[]
length = 1000000
for x in range(len(corpus)):
    tmp=[]
    a1=dic.doc2bow(corpus[x])
    for xx in lda[a1]:
        #print(xx)
        tmp.append(xx[1])
    # if len(tmp)!=lda.num_topics:
    #     continue
    length = min(length, len(tmp))
    d_l.append(tmp)
    labelss.append(labels[x])
new_d_l = []
for i in d_l:
    new_d_l.append(i[: length])

from sklearn.svm  import  SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test =train_test_split(new_d_l,labelss,test_size=0.3)
# sv=SVC(decision_function_shape = 'ovo')
sv = LDA()
# sv = PCA()
print('训练')
sv.fit(X_train,Y_train)
from sklearn.metrics import classification_report

# Y_test = np.array(Y_test).reshape(len(Y_test), -1)
# enc = OneHotEncoder()
# enc.fit(Y_test)
# targets = enc.transform(Y_test).toarray()
print(classification_report(sv.predict(X_test), Y_test))
# sco = sv.score(X_test, Y_test)
# print(sco)
print(sv.predict(X_test))