import os
from ftplib import FTP

# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day1_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day2_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day3_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day4_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day5_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day6_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day7_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day8_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day9_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day10_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day11_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day12_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day13_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day14_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day15_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day16_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day17_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day18_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day19_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day20_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day21_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day22_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day23_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day24_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day25_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day26_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day27_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day28_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day29_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day30_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day31_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day32_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day33_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day34_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day35_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day36_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day37_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day38_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day38_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day39_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day39_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day40_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day40_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day41_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day41_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day42_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day43_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day44_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day44_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day44_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day45_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day45_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day45_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day46_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day46_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day46_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day46_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day46_5.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day46_6.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day46_7.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day46_8.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day47_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day47_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day47_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day47_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day47_5.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day47_6.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day47_7.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day47_8.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day48_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day48_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day48_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day48_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day48_5.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day48_6.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day48_7.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day49_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day49_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day49_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day49_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day50_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day50_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day50_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day50_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day51_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day51_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day51_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day51_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day51_5.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day51_6.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day51_7.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day51_8.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day51_9.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day52_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day52_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day52_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day52_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day53_5.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day53_6.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day54_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day54_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day54_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day54_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day54_5.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day54_6.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day55_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day55_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day55_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day55_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day55_5.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day56_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day56_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day56_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day57_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day57_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day57_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day58_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day58_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day58_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day58_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day58_5.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day58_6.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day59_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day59_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day59_3.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day59_4.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day59_5.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day59_6.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day59_7.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day60_1.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day60_2.gz
# ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day60_3.gz



urls = """
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day52_5.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day52_6.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day53_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day53_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day53_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day53_4.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day60_4.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day60_5.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day60_6.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day60_7.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day61_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day61_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day61_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day61_4.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day61_5.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day61_6.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day61_7.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day61_8.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day62_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day62_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day62_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day62_4.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day62_5.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day62_6.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day62_7.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day62_8.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day62_9.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day62_10.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day63_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day63_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day63_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day63_4.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day64_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day64_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day64_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day65_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day65_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day65_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day65_4.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day65_5.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day65_6.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day65_7.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day65_8.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day65_9.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_4.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_5.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_6.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_7.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_8.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_9.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_10.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day66_11.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day67_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day67_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day67_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day67_4.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day67_5.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day68_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day68_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day68_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day69_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day69_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day69_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day69_4.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day69_5.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day69_6.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day69_7.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day70_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day70_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day70_3.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day71_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day71_2.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day72_1.gz
ftp://ita.ee.lbl.gov/traces/WorldCup/wc_day72_2.gz
"""


def download_file(url):
    ftp_host = 'ita.ee.lbl.gov'
    ftp = FTP(ftp_host)
    ftp.login()

    parts = url.split('/')
    ftp_path = '/'.join(parts[3:])

    local_file_path = os.path.basename(ftp_path)
    with open(local_file_path, 'wb') as local_file:
        ftp.retrbinary('RETR ' + ftp_path, local_file.write)
    ftp.quit()
    # move file to dataset/WorldCup/complete
    os.rename(local_file_path, os.path.basename(f'complete/{local_file_path}'))
    print(f'{url} downloaded to {local_file_path}')

if __name__ == '__main__':
    # with Pool(1) as pool:
    #     pool.map(download_file, urls.split())

    for url in urls.split():
        download_file(url)

# # 先登录到FTP服务器
# ftp_host = 'ita.ee.lbl.gov'
# ftp = FTP(ftp_host)
# ftp.login()
# for url in urls.split():
#     # 解析URL
#
#     parts = url.split('/')
#     # ftp_host = parts[2]
#     ftp_path = '/'.join(parts[3:])
#
#     # # 连接到FTP服务器
#     # ftp = FTP(ftp_host)
#     # ftp.login()  # 匿名登录，如果需要用户名和密码，请在括号中传递用户名和密码
#
#     # 下载文件
#     local_file_path = os.path.basename(ftp_path)
#     with open(local_file_path, 'wb') as local_file:
#         ftp.retrbinary('RETR ' + ftp_path, local_file.write)
#     print(f'{url} downloaded to {local_file_path}')
#
# # 关闭FTP连接
# ftp.quit()
