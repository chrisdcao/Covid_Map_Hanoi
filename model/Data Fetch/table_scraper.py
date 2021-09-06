import requests
from bs4 import BeautifulSoup as bs
import os
import docx2txt

def download(url, file_name):
    response = requests.get(url)
    with open(file_name, "wb") as file:
        file.write(response.content)

def fetch_data_to_txt(url, file_name, dest_folder='../covid_data'):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist
    file_path = dest_folder + '/' + file_name + '.docx'
    download(url, file_path)
    output_txt = dest_folder + '/' + file_name + '.txt'
    with open(output_txt, "w") as text_file:
        print(docx2txt.process(file_path), file=text_file)
    return dest_folder

""" 
Fetch Data: fetching data from the site table to txt, return the directory of data and the last_STT of the patient 

    Input: 
        dest_folder_data: The folder storing data
        dest_folder_updated: The folder which will store the last updated index 
        last_updated_index: The download will search for this in the table and start downloading from it onwards to the latest news. Defaultly set to 0
        url: the url from which will fetch the data from (only from the table)
        parent_abs_link: the link that could be the absolute parent of the relative link inside

    Output: 
        last_udpated_index: the newly updated last_upd_idx
"""

def fetch_data_from_table(dest_folder_data='../covid_data', dest_folder_updated='../covid_data/last_updated', url='https://soyte.hanoi.gov.vn/tin-tuc-su-kien/-/asset_publisher/4IVkx5Jltnbg/content/cap-nhat-tinh-hinh-dich-benh-covid-19-tai-thanh-pho-ha-noi', parent_abs_link='https://soyte.hanoi.gov.vn'):
    r = requests.get(url)
    soup = bs(r.content, 'lxml')

    k = 0
    last_info, child_link, file_name, parent_dir = '','','',''
    for i in soup.select('tbody td:nth-of-type(8)'):
        if i.a:
            # selecting the k-th row of column 2,5,6 (Mã Bệnh nhân, nơi ở, ngày phát hiện) in the table-body tag<> (tbody) element of the html
            file_name = soup.select('tbody td:nth-of-type(2)')[k].text + '%' + soup.select('tbody td:nth-of-type(5)')[k].text.replace(' ','_') + '%' + soup.select('tbody td:nth-of-type(6)')[k].text.replace('/','_')

            # convert the Soup 'tag' object to string, then split at ' ' (which separates each elements in this tag)
            # store all into a_tag_elements[]
            a_tag_elements = str(i.a).split(' ')

            # iterating the array to find element with 'href'
            for elem in a_tag_elements:
                # then take the substring of the relative_link (child) within that 'href' and break
                if 'href' in elem:
                    child_link=elem[elem.find('\"')+1: elem.rfind('\"')]
                    break
            # download link is absolute = parent + child links
            download_link = parent_abs_link + child_link
            
            # fetch the data
            parent_dir = fetch_data_to_txt(download_link, file_name, dest_folder_data)

            # Detect Notification 
            print('Có thông tin. Writing to: {}.txt...'.format(file_name))
        elif i.text == 'Đã hoàn thành điều tra, truy vết':
            print(i.text)
            break
        else:
            last_info = soup.select('tbody td:nth-of-type(1)')[k].text
        k += 1

    print('STT cuối chưa có truy vết: {}. Written to: last_updated.txt'.format(last_info))
    f = open("last_updated.txt", "w")
    f.write(last_info)
    # return the last_updated file and the directory of the file
    #  return last_updated_file

fetch_data_from_table()
