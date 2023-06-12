import os.path

from utils import fileutil
import json
import tqdm


def record_token(key, optlevel):
    os.makedirs(f'../data/tokenset/{key}/{optlevel}', exist_ok=True)

    def file_token(file):
        with open(file, mode='r') as f:
            return f.readlines()

    def split_line(line):
        return line.split()

    files = fileutil.get_files(f'../data/step2/train_114/{key}/{optlevel}')
    token_set = set(
        word
        for file in tqdm.tqdm(files)
        for line in file_token(file)
        for word in split_line(line)
    )

    # 问题代码，怀疑有可能引起dump时出错
    # 这个for循环是清除所有的 \n 字符
    # for token in token_set:
    #     if '\n' in token:
    #         token_set.discard(token)
    #         token_set.add(token.replace('\n', ''))

    # 清除 'pad' 字符串
    token_set.discard('pad')
    token_path = f'../data/tokenset/{key}/{optlevel}/token'
    token_list = list(token_set)
    for token in token_list:
        if token is not str:
            print('token =', token)
    with open(token_path, mode='w') as f:
        json.dump(list(token_set), f)


def legal_token_list(key: str, optlevel):
    """
    @return: 返回恰当的token_set
    """
    token_path = f'../data/tokenset/{key}/{optlevel}/token'
    if not os.path.exists(token_path):
        record_token(key)
    with open(token_path, mode='r') as f:
        token_list = json.load(f)
    return token_list


if __name__ == '__main__':
    keys = [
        'ano',
    ]
    optlevel = 'O0'
    for k in keys:
        record_token(k, optlevel)
