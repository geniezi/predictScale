import cmd
import os
import pwd

# random = 'random'
random = 'random_random'
result_dir = 'randomForest/forest/' + random + '/14_21/'
# 遍历1-7个参数的文件夹
if not os.path.exists(result_dir + 'best/'):
    os.makedirs(result_dir + 'best/')
for i in range(1, 8):
    print(f'正在处理{i}个参数的组合')
    # 每轮进行结果检查，留下最好的几个
    # 读取文件夹下的classification_report.txt文件
    classification_reports = []
    DIR = result_dir + str(i) + '_parameter/'
    folder_list = os.listdir(DIR)
    for folder in folder_list:
        # 排除.DS_Store
        if folder == '.DS_Store':
            continue
        # 打开当前参数的文件夹下的classification_report.txt文件
        with open(DIR + folder + '/classification_report.txt', 'r') as f:
            classification_reports.append((folder, f.read()))
        # for folder, _ in classification_reports:
        #     y=_.split('[')[0].split()[-2]
        #     print(y)
    # 按照f1-score、precision、recall排序
    classification_reports.sort(key=lambda x: float(x[1].split('[')[0].split()[-2]))
    # 留下每种指标最高的
    f1 = classification_reports[-1]
    classification_reports.sort(key=lambda x: float(x[1].split('[')[0].split()[-3]))
    recall = classification_reports[-1]
    classification_reports.sort(key=lambda x: float(x[1].split('[')[0].split()[-4]))
    precision = classification_reports[-1]
    # 判断f1、recall、precision是否相同，相同则放入同一个文件夹，否则分开
    if f1[0] == recall[0] == precision[0]:
        os.system('cp -r ' + DIR + f1[0] + ' ' + result_dir + 'best/'+str(i)+'_f1_recall_precision/')
    elif f1[0] == recall[0]:
        os.system('cp -r ' + DIR + f1[0] + ' ' + result_dir + 'best/'+str(i)+'_f1_recall/')
        os.system('cp -r ' + DIR + precision[0] + ' ' + result_dir + 'best/'+str(i)+'_precision/')
    elif f1[0] == precision[0]:
        os.system('cp -r ' + DIR + f1[0] + ' ' + result_dir + 'best/'+str(i)+'_f1_precision/')
        os.system('cp -r ' + DIR + recall[0] + ' ' + result_dir + 'best/'+str(i)+'_recall/')
    elif recall[0] == precision[0]:
        os.system('cp -r ' + DIR + recall[0] + ' ' + result_dir + 'best/'+str(i)+'_recall_precision/')
        os.system('cp -r ' + DIR + f1[0] + ' ' + result_dir + 'best/'+str(i)+'_f1/')
    else:
        os.system('cp -r ' + DIR + recall[0] + ' ' + result_dir + 'best/'+str(i)+'_recall/')
        os.system('cp -r ' + DIR + precision[0] + ' ' + result_dir + 'best/'+str(i)+'_precision/')
        os.system('cp -r ' + DIR + f1[0] + ' ' + result_dir + 'best/'+str(i)+'_f1/')