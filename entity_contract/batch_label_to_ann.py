from script.txt_to_ann import txt_to_ann
from entity_contract import gen_label

gen_label.run("F:/contract/txt", "F:/contract/label")
txt_to_ann.run("F:/contract", "F:/contract/ann")


