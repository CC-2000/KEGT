"""
常用工具
"""
import json
import time
import os
import yaml

def get_cur_time():
    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    return current_time

def create_folders(file_path):
    """
        递归的创建file_path中所有不存在的文件夹
        file_path可以是文件名也可以是路径
    """
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def parser_set_default(args, **kwargs):
    for k, v in kwargs.items():
        assert hasattr(args, k), f"The parser do not have the attribute {k}"
        setattr(args, k, v)


def print_loud(x, pad=3):
    """
    Prints a string with # box for emphasis.

    Example:
    ############################
    #                          #
    #      DISPLAYED TEXT      #
    #                          #
    ############################
    """

    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print("".join(["#" for _ in range(n + 2 * pad)]))
    
    
def json_data_load(json_data_file_path):
    """
    读取下面这种类型组织的json文件
    {"idx": 0, "questions": ["Pole vault record is held by"], "answers": [" Fabiana Murer"], "subject": ["Q185027"], "relation": ["P1000"], "object": ["Q235118"]}
    {"idx": 1, "questions": ["10,000 metres record is held by"], "answers": [" Kenenisa Bekele", " Kenenisa Bekele Beyecha"], "subject": ["Q163892"], "relation": ["P1000"], "object": ["Q9119"]}
    {"idx": 2, "questions": ["Sergey Bubka record is held by"], "answers": [" pole vault"], "subject": ["Q184556"], "relation": ["P1000"], "object": ["Q185027"]}
    
    return:
        [
            dict_1,
            dict_2,
            dict_3,
            ...
        ]
    """
    with open(json_data_file_path) as fp:
        data = fp.readlines()
    ans = []
    for i in range(len(data)):
        json_data = json.loads(data[i])
        ans.append(json_data)
    return ans

def json_data_dump(json_data_list, save_file_path):
    """_summary_

    Args:
        json_data_list (_type_): 列表，每个元素都是一个json字典
                                        [
                                            dict_1,
                                            dict_2,
                                            dict_3,
                                            ...
                                        ]
    """
    with open(save_file_path, 'w') as fp:
        for json_data in json_data_list:
            json.dump(json_data, fp)
            fp.write('\n')
    fp.close()

def list_data_dump(list_data, save_file_path, sep=' '):
    """
        保存如下形式 [1, 2, 3, 4, 5]
    """
    with open(save_file_path, 'w') as fp:
        for d in list_data:
            fp.write(str(d) + sep)
    fp.close()

def list_data_load(list_data_file_path, sep=' '):
    with open(list_data_file_path) as fp:
        data = fp.readlines()[0]

    data = list(
        map(lambda x: float(x), data.strip().split(sep))
    )
    
    return data

def load_yaml(filename):
    with open(filename, 'r') as f:
        obj = yaml.load(f, Loader=yaml.FullLoader)
    return obj

def save_yaml(filename, obj):
    with open(filename, 'w') as f:
        yaml.dump(obj, f, indent=4, sort_keys=False)
