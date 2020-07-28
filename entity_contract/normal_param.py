START_TAG = "[CLS]"
STOP_TAG = "[SEP]"
tag_dic = {
   "location" : "LOCATION",
    "area" : "AREA",
    "rent" : "RENT",
    "type" : "TYPE",
    "startTerm" : "STARTTERM",
    "endTerm" : "ENDTERM",
    "deadline" : "DEADLINE"
}
dic_path = ""

labels = {"LOCATION","AREA","RENT","STARTTERM","ENDTERM", "TYPE", "DEADLINE"}

EMBEDDING_DIM = 5
HIDDEN_DIM = 4