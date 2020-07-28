import jiagu
import jieba
#jiagu.init() # 可手动初始化，也可以动态初始化

text = '自租赁甲方门市房起，具有门市房使用权，乙方需装修房屋，如改变内部房屋格局，须经甲方同意后方可改变房屋格局，装修费由乙方自担。但租赁期满如不再租赁，须经甲方检查房屋，如发现房屋在乙方租赁期间主体结构遭到，须按市价进行补偿。'


words = jiagu.seg(text) # 分词
print(words)
cut_words = jieba.cut(text)
seg_str = "/".join(list(cut_words))
print(seg_str)

pos = jiagu.pos(words) # 词性标注
print(pos)

ner = jiagu.ner(words) # 命名实体识别
print(ner)

# text = '姚明1980年9月12日出生于上海市徐汇区，祖籍江苏省苏州市吴江区震泽镇，前中国职业篮球运动员，司职中锋，现任中职联公司董事长兼总经理。'
knowledge = jiagu.knowledge(text)
print(knowledge)

# import jiagu

text = '很讨厌还是个懒鬼'
sentiment = jiagu.sentiment(text)
print(sentiment)

