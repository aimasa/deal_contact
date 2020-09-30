from spyder import www_64365_com as spy64365
from spyder import www_chinalawedu_com as spychina
from spyder import www_diyifanwen_com as spydiyifan
import os
from flask import Flask,render_template,request,jsonify,url_for
def match(network_name, kind):
    if network_name is "64359" and kind is "劳动":
        return "https://www.64365.com/contract/lwht/", "劳动合同"
    if network_name is "chinalawedu" and kind is "借款":
        return "http://www.chinalawedu.com/web/192/page", "借款合同"
    if network_name is "diyifanwen" and kind is "劳动":
        return "https://www.diyifanwen.com/fanwen/laodonghetong/", "劳动合同"
def download(name, url, path):
    if name is "64359":
        spy64365.run(first_url=url, document_path_header=path)
    if name is "chinalawedu":
        spychina.run(url, path)
    if name is "diyifanwen":
        spydiyifan.run(url, path)
def run(name, kind_of_contract, save_file_head):
    url, file_name = match(name, kind_of_contract)
    path = os.path.join(save_file_head, file_name)
    os.makedirs(path)
    download(name, url, path)
app = Flask(__name__)
@app.route('/spyder/contact', methods = ["POST"])
def contact_spyder():
    if request.method == "POST":
        name = request.form["name"]
        kind_of_contract = request.form["save_file_head"]
        save_file_head = request.form["save_file_head"]
        run(name, kind_of_contract, save_file_head)
@app.route('/spyder/contact', methods = ["POST"])
def comic_spyder():
    if request.method == "POST":
        comic_chinese_name = request.form["comic_chinese_name"]
        comic_name = request.form["comic_name"]
        series_id_first = request.form["series_id_first"]
        series_id_last = request.form["series_id_last"]
        zip_type = request.form["zip_type"]
        folder_name_header = request.form["folder_name_header"]
        access_token = request.form["access_token"]
        comic_id = request.form["comic_id"]
        lezhin_cookie = request.form["lezhin_cookie"]
        spare_time = request.form["spare_time"]
        info = config(comic_chinese_name, comic_name, series_id_first, series_id_last, zip_type, folder_name_header, access_token, comic_id, lezhin_cookie, spare_time)

class config(object):
    def __init__(self, comic_chinese_name, comic_name, series_id_first, series_id_last, zip_type, folder_name_header, access_token, comic_id, lezhin_cookie, spare_time):
        self.comic_chinese_name = comic_chinese_name
        self.comic_name = comic_name
        self.series_id_first = series_id_first
        self.series_id_last = series_id_last
        self.zip_type = zip_type
        self.folder_name_header = folder_name_header
        self.access_token = access_token
        self.comic_id = comic_id
        self.lezhin_cookie = lezhin_cookie
        self.spare_time = spare_time





