
from pyquery import PyQuery as pq


import urllib.request as ulb
# 下载地址的前缀域名
based_url = "https://www.64365.com/"
# 对目录页的html进行解析，获得目录中的合同下载链接同时获取合同的名字
def read_html(url):
    doc = pq(url, headers={'User-Agent': 'Mozilla/5.0 '
                                           '(Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                                           '(KHTML, like Gecko) Chrome/46.0.2490.71 Safari/537.36'},encoding='utf-8')
    url_list = ["https:"+link.attr('href') for link in doc.items('.hetong-list li a[href^="//download"]')]
    url_name_list = [link_name.text() for link_name in doc.items('.hetong-list li h3')]

    # new_url = list(filter(lambda x: re.search('download', x) != None, url_list))

    return url_list, url_name_list
# 获得下一页目录地址
def get_last_url(url_current):

    doc = pq(url_current, headers={'User-Agent': 'Mozilla/5.0 '
                                           '(Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                                           '(KHTML, like Gecko) Chrome/46.0.2490.71 Safari/537.36'},encoding='utf-8')
    url_last = [based_url + link_last.attr('href') for link_last in doc.items('.u-page a:nth-last-child(1)')]
    # for link_temp in doc.items('.u-page .borl.u-p-on'):
    #     print(link_temp.text())
    page_number = [link_number.text() for link_number in doc.items('.u-page .borl.u-p-on')]
    if url_last[0] == url_current:
        print("已经完成" + page_number[0] + "页的打印")
        return None

    print("已经打印完成第" + page_number[0] + "页，准备打印下一页")
    return url_last[0]





# def check_path(doc_path):


# 对合同下载地址和合同名字进行匹配并且下载文件
def download_url(download_url,document_name,download_path_header):
    if download_path_header is None or len(download_path_header) == 0:
        document_path = check_file_name(document_name + ".doc")
    else:
        document_path = download_path_header + "/" + check_file_name(document_name + ".doc")
    response = ulb.Request(download_url,headers={'User-Agent': 'Mozilla/5.0 '
                                           '(Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                                           '(KHTML, like Gecko) Chrome/46.0.2490.71 Safari/537.36'})
    contract_document = ulb.urlopen(response).read()
    save_document(document_path,contract_document)

# 将下载好的文件保存到本地
def save_document(document_path,contract_document):
    with open(document_path,'wb') as f:
        f.write(contract_document)

# 确定合同名字和合同下载链接匹配正确与否
def check_len(url_list, url_name_list):
    if len(url_list) == len(url_name_list):
        return True
    return False
# 根据传入的合同进行下载
def start_download(first_url,document_path_header):
    url_list,url_name_list = read_html(first_url)
    if(check_len(url_list, url_name_list)):
        for url_index in range(len(url_list)):
            print("开始下载本页第" + str(url_index) + "个合同")
            try:
                download_url(url_list[url_index], url_name_list[url_index],document_path_header)
            except Exception as e:
                print(e)
                continue



# 对合同名进行容错
def check_file_name(file_name):
    file_name = file_name.replace("/","_")
    file_name = file_name.replace("?","_")
    return file_name


# 爬虫运行下载
def run(first_url,document_path_header):
    start_download(first_url,document_path_header)
    if(get_last_url(first_url) !=None):
        run(get_last_url(first_url),document_path_header)


if __name__ == "__main__":
    # get_download_url("https:" + "//download.64365.com/contact/default.aspx?id=751812&&type=1",'个人租房合同范本')
    url = "https://www.64365.com/contract/lwht/"
    file_header = "G:/律师-劳动"
    run(url,file_header)