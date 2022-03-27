# -*- coding:utf-8 -*-
import copy
import json
# from matplotlib import pyplot
import random
import matplotlib.pyplot as plt
import random

import numpy as np
import matplotlib.pyplot as plt

def dict_to_json(dict_temp, file_name):
    print("开始写入文件")
    print(file_name)
    with open(file_name, "w", encoding='utf-8') as f:
        json.dump(dict_temp, f, indent=2, ensure_ascii=False)
    print("结束写入文件")


def json_to_dict(file_path):
    print("开始读取文件")
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    print("结束读取文件")


def get_another_json(file_path, file_path2):
    dict_all = json_to_dict(file_path)
    list_t = []
    # str_replace = "附件：\n                    \n                    |\n                    \n                    |\n                    \n                    |\n                    \n                主办：天津市人民政府办公厅 版权所有© 承办：天津市人民政府办公厅政务信息发布中心\n                    网站标识码：1200000052 \n                    \n                \n                    系统检测到您正在使用ie8以下内核的浏览器，不能实现完美体验，请更换或升级浏览器访问！\n                \n\n                \n\n                |\n\n                \n\n                |\n\n                \n\n                |\n\n                \n\n\n\n            主办：天津市人民政府办公厅 版权所有©承办：天津市人民政府办公厅政务信息发布中心网站标识码：1200000052 \n                        \n"
    for i in dict_all:
        dict_temp = copy.deepcopy(dict_all[i])
        dict_temp["document"] = dict_temp["document"]
        # dict_temp["document"] = dict_temp["document"].replace(str_replace, "")
        list_t.append(dict_temp)
    random.shuffle(list_t)
    dict_return = {}
    for i in list_t:
        dict_return[i["query"]] = i
    dict_to_json(dict_return, file_path2)


def get_dict_from_txt(txt_path):
    dict_return = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        line_list = f.readlines()
        for i in line_list:
            dict_i = json.loads(i)
            category = dict_i['category']
            if category in dict_return:
                dict_return[category] += 1
            else:
                dict_return[category] = 1
    print(len(dict_return))
    dict_to_json(dict_return, 'all_category.json')


def get_json_by_category(category, txt_path):
    dict_return = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        line_list = f.readlines()
        for index, i in enumerate(line_list):
            dict_i = {}

            # i = i.decode('GBK').encode('utf-8')
            dict_i = json.loads(i)

            category_i = dict_i['category']
            if category_i == category:
                qid = dict_i["qid"]
                # if qid == "qid_5066449769491887589":
                #     continue
                dict_return[qid] = dict_i
    print(len(dict_return))
    dict_to_json(dict_return, str(len(dict_return)) + "category" + category + ".json")


def get_data2(content_dict):
    """
    获取模型需要的数据
    :param content_dict: 每个类型的数据字典
    :return: query, document
    """
    dict_all = json_to_dict(content_dict)
    query = []
    document = []
    for id_i in dict_all:
        dict_i = dict_all[id_i]
        title = dict_i['title']
        desc = dict_i['desc']
        answer = dict_i['answer']
        if desc == "":
            if len(title) > 100:
                title = title[:100]
            query.append(title)
        else:
            if len(desc) > 100:
                desc = desc[:100]
            query.append(desc)
        if len(answer) > 400:
            answer = answer[:400]
        document.append(answer)

    # length = len(query)
    # labels = np.eye(length)

    return query, document

def get_data3(content_dict):
    """
    获取模型需要的数据
    :param content_dict: 每个类型的数据字典
    :return: query, document
    """
    dict_all = json_to_dict(content_dict)
    query = []
    document = []
    for id_i in dict_all:
        dict_i = dict_all[id_i]
        title = dict_i['similar_q']
        answer = dict_i['d']

        if len(title) > 100:
            title = title[:100]
        query.append(title)

        if len(answer) > 400:
            answer = answer[:400]
        document.append(answer)

    # length = len(query)
    # labels = np.eye(length)

    return query, document


def get_data4(json_file, keys):
    """
    read divided_tripletData_removeRedundant.json data
    :param dict_all:
    :param keys:
    :return:
    """
    dict_all = json_to_dict(json_file)
    dict_temp = dict_all[keys]
    query = []
    document = []
    for id_i in dict_temp:
        dict_i = dict_temp[id_i]
        title = dict_i['query']
        answer = dict_i['document']

        if len(title) > 100:
            title = title[:100]
        query.append(title)

        if len(answer) > 400:
            answer = answer[:400]
        document.append(answer)
    return query, document

def get_data6(json_file):
    """
    read divided_tripletData_removeRedundant.json data
    :param dict_all:
    :param keys:
    :return:
    """
    dict_all = json_to_dict(json_file)
    query = []
    document = []
    for id_i in dict_all:
        dict_i = dict_all[id_i]
        title = dict_i['query']
        answer = dict_i['document']

        if len(title) > 100:
            title = title[:100]
        query.append(title)

        if len(answer) > 400:
            answer = answer[:400]
        document.append(answer)
    return query, document

def get_data5(json_file):
    """
    read divided_tripletData_removeRedundant.json data
    :param dict_all:
    :param keys:
    :return:
    """
    dict_temp = json_to_dict(json_file)
    query = []
    document = []
    for id_i in dict_temp:
        dict_i = dict_temp[id_i]
        title = dict_i['query']
        answer = dict_i['document']

        if len(title) > 100:
            title = title[:100]
        query.append(title)

        if len(answer) > 400:
            answer = answer[:400]
        answer = title + answer
        document.append(answer)
    return query, document

def get_data6(json_file):
    """
    read divided_tripletData_removeRedundant.json data
    :param dict_all:
    :param keys:
    :return:
    """
    dict_temp = json_to_dict(json_file)
    query = []
    document = []
    for id_i in dict_temp:
        dict_i = dict_temp[id_i]
        title = dict_i['title']
        fulltext = dict_i['fulltext']
        standard = dict_i['standard']


        if len(title) > 100:
            title = title[:100]
        query.append(title)

        if len(fulltext) > 400:
            fulltext = fulltext[:400]
        if len(standard) > 400:
            standard = standard[:400]

        answer = title + fulltext + standard
        document.append(answer)
    return query, document


def write_txt_result(txt_file, str_log):
    with open(txt_file, "a", encoding="utf-8") as f:
        f.write(str_log + "\n")


def drawScore(out, label, name):

    plt.scatter(out, label)
    plt.xlabel('Score')
    plt.ylabel('Label')
    plt.title(name)
    plt.xlim(0, 1)
    plt.ylim(-0.5, 1.5)
    plt.show()


def drawSocre2(x, y, name, xname, x1, x2, yname, y1, y2):
    plt.scatter(x, y)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(name)
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.show()


def drawScore3(x1, y1, x2, y2, name, xname, yname, xmin, xmax, ymin, ymax):
    plt.xlabel(xname)

    plt.ylabel(yname)

    plt.title(name)

    plt.xlim(xmax=xmax, xmin=xmin)

    plt.ylim(ymax=ymax, ymin=ymin)

    # 画两条（0-9）的坐标轴并设置轴标签x，y

    # x1 = np.random.normal(2, 1.2, 300)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
    #
    # y1 = np.random.normal(2, 1.2, 300)  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
    #
    # x2 = np.random.normal(7.5, 1.2, 300)
    #
    # y2 = np.random.normal(7.5, 1.2, 300)

    colors1 = '#00CED1'  # 点的颜色

    colors2 = '#DC143C'

    area = np.pi * 1 ** 2  # 点面积

    # 画散点图

    plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='label 0')

    plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='label 1')

    # plt.plot([0, 9.5], [9.5, 0], linewidth='0.5', color='#000000')

    plt.legend()

    # plt.savefig(r'C:\Users\jichao\Desktop\大论文\12345svm.png', dpi=300)

    plt.show()



if __name__ == '__main__':
    # file_path1 = './baike_data/baike_qa_valid.json'
    # file_path2 = './baike_data/baike_qa_train.json'
    # # get_dict_from_txt(file_path2)
    # category1 = "社会民生-公务办理"
    # get_json_by_category(category1, file_path2)
    print("main")
    # file_name = "./2778category社会民生-公务办理.json"
    # all_query, all_document = get_data2(file_name)
    # query_len = []
    # document_len = []
    # for i in all_query:
    #     query_len.append(len(i))
    # for i in all_document:
    #     document_len.append(len(i))
    # drawHist(query_len)

    # get_another_json("2778category社会民生-公务办理.json", "society1.json")


    # N = 40
    # x = np.random.rand(N)
    # y = np.random.rand(N) * 10
    #
    # # random colour for points, vector of length N
    # colors = np.random.rand(N)
    #
    # # area of the circle, vectoe of length N
    # area = (30 * np.random.rand(N)) ** 2
    # # 0 to 15 point radii
    #
    # # a normal scatter plot with default features
    # plt.scatter(x, y, alpha=0.8)
    # plt.xlabel('Numbers')
    # plt.ylabel('Values')
    # plt.title('Normal Scatter Plot')
    # plt.show()
    #
    # # a scater plot with different size
    # plt.figure()
    # plt.scatter(x, y, s=area, alpha=0.8)
    # plt.xlabel('Numbers')
    # plt.ylabel('Values')
    # plt.title('Different Size')
    # plt.show()
    #
    # # a scatter plot with different collour
    # plt.figure()
    # plt.scatter(x, y, c=colors, alpha=0.8)
    # plt.xlabel('Numbers')
    # plt.ylabel('Values')
    # plt.title('Different Colour')
    # plt.show()
    #
    # # A combined Scatter Plot
    # plt.figure()
    # plt.scatter(x, y, s=area, c=colors, alpha=0.8)
    # plt.xlabel('Numbers')
    # plt.ylabel('Values')
    # plt.title('Combined')
    # plt.show()



    # file_name = "./society1.json"
    # all_query, all_document = get_data2(file_name)
    # sample_list = [i for i in range(len(all_query))]
    # tmp_list = random.sample(sample_list, 150)  # 随机选取出了 [3, 4, 2, 0]
    # print(tmp_list)
    # train_query_t = [all_query[i] for i in tmp_list]  # ['d', 'e', 'c', 'a']
    # train_document_t = [all_document[i] for i in tmp_list]  # [3, 4, 2, 0]
    # dict_return = {}
    # for i in range(len(tmp_list)):
    #     dict_return[i] = {"q": train_query_t[i], "d": train_document_t[i], "similar_q": ""}
    # dict_to_json(dict_return, "待修改150.json")

    # file_name = "./combine_all.json"
    # file_name2 = "./combine_all2.json"
    # get_another_json(file_name, file_name2)
    # print(len(json_to_dict(file_name2)))
    dict_ = json_to_dict("similary_query/wpy_2_20_v2_2.json")
    print(len(dict_))






