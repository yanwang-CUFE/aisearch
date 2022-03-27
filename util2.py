from utils import json_to_dict, dict_to_json
import random
import copy

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

def get_another_json2(file_path, file_path2):
    dict_all = json_to_dict(file_path)
    list_t = []
    # str_replace = "附件：\n                    \n                    |\n                    \n                    |\n                    \n                    |\n                    \n                主办：天津市人民政府办公厅 版权所有© 承办：天津市人民政府办公厅政务信息发布中心\n                    网站标识码：1200000052 \n                    \n                \n                    系统检测到您正在使用ie8以下内核的浏览器，不能实现完美体验，请更换或升级浏览器访问！\n                \n\n                \n\n                |\n\n                \n\n                |\n\n                \n\n                |\n\n                \n\n\n\n            主办：天津市人民政府办公厅 版权所有©承办：天津市人民政府办公厅政务信息发布中心网站标识码：1200000052 \n                        \n"
    for i in dict_all:
        dict_temp = copy.deepcopy(dict_all[i])
        # dict_temp["document"] = dict_temp["document"]
        # dict_temp["document"] = dict_temp["document"].replace(str_replace, "")
        list_t.append(dict_temp)
    random.shuffle(list_t)
    dict_return = {}
    for i in list_t:
        dict_return[i["title"]] = i
    dict_to_json(dict_return, file_path2)

if __name__ == "__main__":
    # file_name = "./combine_all.json"
    # file_name2 = "similary_query/combine_all2.json"
    # get_another_json(file_name, file_name2)
    # print(len(json_to_dict(file_name2)))
    file_name = "./wpy_2_20_v2.json"
    file_name2 = "similary_query/wpy_2_20_v2_2.json"
    get_another_json2(file_name, file_name2)
    print(len(json_to_dict(file_name2)))