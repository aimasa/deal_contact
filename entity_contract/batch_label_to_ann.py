from script.txt_to_ann import txt_to_ann
from entity_contract import gen_label
'''
用于将txt预测得到对应的label并将其转换成ann格式。
'''
gen_label.run("F:/contract/txt", "F:/contract/label")
txt_to_ann.run("F:/contract", "F:/contract/ann")


