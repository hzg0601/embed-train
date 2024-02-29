import json
from typing import List


def json_merge(file_names: List[str], saved_file_path):
    # 初始化一个空字典来收集所有数据
    merged_data = {}

    # 遍历文件名列表，加载每个文件的内容并合并
    for file_name in file_names:
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
            merged_data.update(data)  # 假设每个文件顶层是一个字典

    # 将合并后的JSON数据转换为格式化的字符串，提高可读性
    readable_json = json.dumps(merged_data, indent=4, ensure_ascii=False)

    # 将合并后的格式化JSON数据保存到新文件
    with open(saved_file_path, 'w', encoding='utf-8') as new_file:
        new_file.write(readable_json)


def json_convert_readable(file_path, saved_file_path):
    # 读取原始JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 将JSON数据转换为格式化的字符串，提高可读性
    readable_json = json.dumps(data, indent=4, ensure_ascii=False)

    # 将格式化的JSON字符串保存到新文件
    with open(saved_file_path, 'w', encoding='utf-8') as new_file:
        new_file.write(readable_json)
