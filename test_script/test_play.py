from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('D:/stanford-corenlp-full-2016-10-31', lang='zh')#处理中文需指定lang='zh',英文不用，其它语言也有
sentence = '自租赁甲方门市房起，具有门市房使用权，乙方需装修房屋，如改变内部房屋格局，须经甲方同意后方可改变房屋格局，装修费由乙方自担。但租赁期满如不再租赁，须经甲方检查房屋，如发现房屋在乙方租赁期间主体结构遭到，须按市价进行补偿。'

# sentence = '我是tab，我现在正在学习python'
#分词
print(nlp.word_tokenize(sentence))
#词性标注
print(nlp.pos_tag(sentence))
#命名实体识别
print(nlp.ner(sentence))

import jiagu
import jieba
#jiagu.init() # 可手动初始化，也可以动态初始化

text = '自租赁甲方门市房起，具有门市房使用权，乙方需装修房屋，如改变内部房屋格局，须经甲方同意后方可改变房屋格局，装修费由乙方自担。但租赁期满如不再租赁，须经甲方检查房屋，如发现房屋在乙方租赁期间主体结构遭到，须按市价进行补偿。'


words = jiagu.seg(text) # 分词
print(words)
cut_words = jieba.cut(text)
seg_str = "/".join(list(cut_words))
print(seg_str)