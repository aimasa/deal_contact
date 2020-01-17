from docx import Document
def read_para(path):
    document = Document(path)
    result = document.paragraphs
    print()

if __name__=="__main__":
    read_para("F:/rent_house_contract_pos/律师-借款/安徽农民工劳动合同.docx")
