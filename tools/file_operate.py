import os
import glob


class FileOperate():

    # 遍历文件夹中所有file_extension后缀的文件
    @staticmethod
    def find_files(folder_path, file_extension):
        all_files = []
        for file in glob.glob(os.path.join(folder_path, "*" + file_extension)):
            all_files.append(file)
        return all_files
    
    @staticmethod
    def find_all_files(folder_path):
        all_files = []
        for file in glob.glob(os.path.join(folder_path, "*")):
            all_files.append(file)
        return all_files


def test_find_json_files():
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "..", "data","datasets")
    json_files = FileOperate.find_files(folder_path, ".json")
    for file in json_files:
        print(file)


if __name__ == '__main__':  # sourcery skip: raise-specific-error
    test_find_json_files()
    pass
