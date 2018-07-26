# -*- coding: utf-8 -*-

# 标签提取模块 pyhton脚本, cmd调用方法： python tagging.py --dir='xxx'  #xxx为json文件集所在的目录
# 可能需要修改的地方：NBA专用名词的词典的路径('../NBAdict.txt')，
#     停用词表的路径('../stopword.txt')，日志文件路径('../tagging_log.txt')
#
# 所需库：jieba-fast, json, random, os, logging
# 2018-7-20

import logging
import os
file_handler = logging.FileHandler("tagging_log.txt")   # 将日志打印到文件中
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.basicConfig(handlers=[file_handler], level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)    # 日志对象

# consolo_handler = logging.StreamHandler()   # 将日志打印到控制台中
# consolo_handler.setLevel(logging.DEBUG)
# logger.addHandler(consolo_handler)


def tag_one_file_test(file: str):
    '''
    测试版，在控制台输出提取的标签。
    内部需要：NBA专用名词的词典txt的路径，停用词表txt的路径；
    :param file: 一个json文件的路径，键为 title、url、reply、views、comefrom、time、text、tags
    :return:
    '''
    import json
    import jieba_fast as jb
    jb.load_userdict('dict3.txt')   #专用字典
    import jieba_fast.analyse as jbana
    textranker = jbana.TextRank()
    tfidfer = jbana.TFIDF()   # 分词器
    textranker.set_stop_words('stop.txt')
    tfidfer.set_stop_words('stop.txt')  # stop words

    try:
        json_file = open(file, 'r', encoding='utf-8')
    except IOError:
        print('fail to open', file, IOError, sep=' ')
        return
    doc_list = json.load(json_file)
    for doc in doc_list:
        title = doc['title']
        text = doc['text']
        true_tag = doc['tags'] if 'tags' in doc.keys() else None
        whole = title + ' '
        for string in text:
            whole = whole + string  # 整个文章拼在一起

        split_gen = jb.cut(whole)   # 是生成器
        split_whole = ''
        for s in split_gen:
            split_whole = split_whole + ' ' + s     # 分词后的文章

        tag1w = tfidfer.extract_tags(split_whole, topK=10, withWeight=True
                                 , allowPOS=('nr', 'nz')
                                 )
        tag2w = tfidfer.extract_tags(split_whole, topK=10, withWeight=True
                                  , allowPOS=('an', 'b', 'j',  'l', 'Ng',
                                              'n', 'nr', 'ns', 'nz', 'nt')
                                 )
        tag3w = textranker.textrank(split_whole, topK=10, withWeight=True
                              # , allowPOS=('nr', 'nz', 'n')
            #                   , allowPOS=(
            # 'Ag', 'a', 'ad', 'an', 'b', 'c', 'dg', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'Ng',
            # 'n', 'nr', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 's', 'tg', 't', 'u'
            # , 'vg', 'v', 'vd', 'vn', 'w', 'x', 'y', 'z', 'un')  # all pos
            #                   , allowPOS=(
            # 'Ag', 'a', 'ad', 'an', 'b', 'c', 'dg', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'Ng',
            # 'n', 'nr', 'ns', 'nt', 'nz', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'y', 'z', 'un')
                              , allowPOS=('an', 'b', 'j', 'k', 'l', 'Ng',
                                          'n', 'nr', 'ns', 'nz', 'vn', 's')
                              )
        tag1 = []
        wei1 = []
        for tup in tag1w:
            tag1.append(tup[0])
            wei1.append(tup[1])
        tag2 = []
        wei2 = []
        for tup in tag2w:
            tag2.append(tup[0])
            wei2.append(tup[1])
        tag3 = []
        wei3 = []
        for tup in tag3w:
            tag3.append(tup[0])
            wei3.append(tup[1])
#: 下面是让3个提取器投票选出最后的关键词
# 方案1:
        # final_tag = set()
        # import random
        # choose_size = 5
        # if len(tag1) < choose_size:
        #     final_tag = final_tag.union(set(tag1))
        # else:
        #     cho1 = random.choices(tag1, weights=wei1, k=choose_size)
        #     final_tag = final_tag.union(set(cho1))
        #
        # if len(tag2) < choose_size:
        #     final_tag = final_tag.union(set(tag2))
        # else:
        #     cho2 = random.choices(tag2, weights=wei2, k=choose_size)
        #     final_tag = final_tag.union(set(cho2))
        #
        # if len(tag3) < choose_size:
        #     final_tag = final_tag.union(set(tag3))
        # else:
        #     cho3 = random.choices(tag3, weights=wei3, k=choose_size)
        #     final_tag = final_tag.union(set(cho3))
# 方案2:
        final_tag = list()
        import random
        choose_size = 5     # 一个参数，控制随机选择时选择的数目
        if len(tag1) < choose_size:
            final_tag.extend(tag1w)
        else:
            cho1 = random.choices(tag1w, weights=wei1, k=choose_size)   # 根据分词器判断的权重加权随机选择choose_size个关键词
            final_tag.extend(cho1)      # 合并每个提取器的提议

        if len(tag2) < choose_size:
            final_tag.extend(tag2w)
        else:
            cho2 = random.choices(tag2w, weights=wei2, k=choose_size)
            final_tag.extend(cho2)

        if len(tag3) < choose_size:
            final_tag.extend(tag3w)
        else:
            cho3 = random.choices(tag3w, weights=wei3, k=choose_size)
            final_tag.extend(cho3)

        final_tag.sort(key=lambda x: float(x[1]), reverse=True)     # 所有提议的关键词按权重排序（有重复）
        tag = []
        wei = []
        for tup in final_tag:
            tag.append(tup[0])
            wei.append(tup[1])
        if len(tag) < 2 * choose_size:
            final_tag = tag
        else:
            choose = random.choices(tag, weights=wei, k=2*choose_size)
            final_tag = choose   # 再做一次按权重的随机抽取，大小为2*choose_size
        final_tag = set(final_tag)      # 去重
        print('get tags:', final_tag)
        # 这里还可以考虑如下修改:先去重保持降序性，不过现在的结果似乎还可以，甚至比排序的更好些
        # final_tag = set(final_tag)
        # final_tag = sorted(final_tag, key=lambda x: float(x[1]), reverse=True)     # 关键词按权重排序
        # tag = []
        # wei = []
        # for tup in final_tag:
        #     tag.append(tup[0])
        #     wei.append(tup[1])
        # if len(tag) < 2 * choose_size:
        #     final_tag = tag
        # else:
        #     choose = random.choices(tag, weights=wei, k=2*choose_size)
        #     final_tag = choose
    json_file.close()


def tag_one_file(file: str):
    '''
    服务器版：为一个json文件提取标签，在控制台输出 增加标签后的一整个json文件。
    内部需要（可固定）：NBA专用名词的词典的路径('../NBAdict.txt')，停用词表的路径('../stopword.txt')，
                        设置choose_size参数,控制随机采纳分词的提议时的选择的数目
    :param file: 一个json文件的路径，数组，每个数组单元为一篇新闻报道，其键为 title、url、reply、views、comefrom、time、text、tags
    :return:打印增加标签后的json文件
    '''
    import json
    import jieba as jb0
    jb0.setLogLevel(logging.INFO)
    import jieba_fast as jb
    jb.setLogLevel(logging.INFO)    # 让jieba不输出debug信息

    try:
        jb.load_userdict('NBAdict.txt')   # 专用字典
    except:
        logger.exception('fail to open dictionary')

    from jieba_fast.analyse.textrank import TextRank
    from jieba_fast.analyse.tfidf import TFIDF

    textranker = TextRank()
    tfidfer = TFIDF()   # 分词器
    try:
        textranker.set_stop_words('stopword.txt')
        tfidfer.set_stop_words('stopword.txt')  # stop words
    except:
        logger.exception('fail to set stop words')

    try:
        json_file = open(file, 'r', encoding='utf-8')
    except IOError as ioe:
        logger.exception('fail to open ' + file)  # 打开文件失败
        raise ioe
    try:
        doc_list = json.load(json_file)
    except Exception as e:  # 不知道error类型
        logger.exception('fail to load json file:' + file)  # 打开文件失败
        raise e

    for doc in doc_list:    # 对每一篇报道
        keys = doc.keys()
        title = doc['title'] if 'title' in keys else ''
        text = doc['text'] if 'text' in keys else ''
        old_tag = doc['tags'] if 'tags' in keys else ''     # 原来爬到的tags
        whole = title + ' '
        for string in text:     # 迭代器：为空的话在for循环中也不会报错的
            whole = whole + string  # 整个文章拼在一起

        split_gen = jb.cut(whole)   # 按字典分词，是生成器
        split_whole = ''
        for s in split_gen:
            split_whole = split_whole + ' ' + s     # 分词后的文章
        # 3个分词器，每个提议的结果都是(tag,weight)形式的list
        tag1w = tfidfer.extract_tags(split_whole, topK=10, withWeight=True, allowPOS=('nr', 'nz'))
        tag2w = tfidfer.extract_tags(split_whole, topK=10, withWeight=True,
                                     allowPOS=('an', 'b', 'j',  'l', 'Ng', 'n', 'nr', 'ns', 'nz', 'nt'))
        tag3w = textranker.textrank(split_whole, topK=10, withWeight=True,
                                    allowPOS=('an', 'b', 'j', 'k', 'l', 'Ng', 'n', 'nr', 'ns', 'nz', 'vn', 's'))
        #: 将所有的权重值提取出来形成一个list用于后面的随机选择
        # tag1 = []
        wei1 = []
        for tup in tag1w:
            # tag1.append(tup[0])
            wei1.append(tup[1])
        # tag2 = []
        wei2 = []
        for tup in tag2w:
            # tag2.append(tup[0])
            wei2.append(tup[1])
        # tag3 = []
        wei3 = []
        for tup in tag3w:
            # tag3.append(tup[0])
            wei3.append(tup[1])

        final_tagw = list()
        import random
        choose_size = 5     # 一个参数，控制随机选择时选择的数目
        if len(tag1w) < choose_size:
            final_tagw.extend(tag1w)
        else:
            cho1 = random.choices(tag1w, weights=wei1, k=choose_size)   # 根据分词器判断的权重加权随机选择choose_size个关键词
            final_tagw.extend(cho1)      # 合并每个提取器的提议

        if len(tag2w) < choose_size:
            final_tagw.extend(tag2w)
        else:
            cho2 = random.choices(tag2w, weights=wei2, k=choose_size)
            final_tagw.extend(cho2)

        if len(tag3w) < choose_size:
            final_tagw.extend(tag3w)
        else:
            cho3 = random.choices(tag3w, weights=wei3, k=choose_size)
            final_tagw.extend(cho3)

        final_tagw.sort(key=lambda x: float(x[1]), reverse=True)     # 所有提议的关键词按权重排序（有重复）
        tag = []
        wei = []
        for tup in final_tagw:
            tag.append(tup[0])
            wei.append(tup[1])

        final_tag = list()
        if len(tag) < 2 * choose_size:
            final_tag = tag
        else:
            choose = random.choices(tag, weights=wei, k=2*choose_size)
            final_tag = choose   # 再做一次按权重的随机抽取，大小为2*choose_size
        final_tag = set(final_tag + old_tag)      # 合并原标签并去重

        doc['tags'] = list(final_tag)   # 修改原标签
    print(doc_list)
    # 打印出整个json，不写入文件
    json_file.close()


def tag_one_file2(file: str):
    '''
    服务器版：为一个json文件提取标签，在输出 增加标签后的一整个json文件 到原文件。
    内部需要（可固定）：NBA专用名词的词典的路径('../NBAdict.txt')，停用词表的路径('../stopword.txt')，
                        设置choose_size参数,控制随机采纳分词的提议时的选择的数目
    :param file: 一个json文件的路径，数组，每个数组单元为一篇新闻报道，其键为 title、url、reply、views、comefrom、time、text、tags
    :return:打印增加标签后的json文件
    '''
    import json
    import jieba as jb0
    jb0.setLogLevel(logging.INFO)
    import jieba_fast as jb
    jb.setLogLevel(logging.INFO)    # 让jieba不输出debug信息

    try:
        jb.load_userdict('NBAdict.txt')   # 专用字典
    except:
        logger.exception('fail to open dictionary')

    from jieba_fast.analyse.textrank import TextRank
    from jieba_fast.analyse.tfidf import TFIDF

    textranker = TextRank()
    tfidfer = TFIDF()   # 分词器
    try:
        textranker.set_stop_words('stopword.txt')
        tfidfer.set_stop_words('stopword.txt')  # stop words
    except:
        logger.exception('fail to set stop words')

    try:
        json_file = open(file, 'r', encoding='utf-8')
    except IOError as ioe:
        logger.exception('fail to open ' + file)  # 打开文件失败
        raise ioe
    try:
        doc_list = json.load(json_file)
    except Exception as e:  # 不知道error类型
        logger.exception('fail to load json file:' + file)  # 打开文件失败
        json_file.close()
        raise e

    for doc in doc_list:    # 对每一篇报道
        keys = doc.keys()
        title = doc['title'] if 'title' in keys else ''
        text = doc['text'] if 'text' in keys else ''
        old_tag = doc['tags'] if 'tags' in keys else ''     # 原来爬到的tags
        whole = title + ' '
        for string in text:     # 迭代器：为空的话在for循环中也不会报错的
            whole = whole + string  # 整个文章拼在一起

        split_gen = jb.cut(whole)   # 按字典分词，是生成器
        split_whole = ''
        for s in split_gen:
            split_whole = split_whole + ' ' + s     # 分词后的文章
        # 3个分词器，每个提议的结果都是(tag,weight)形式的list
        tag1w = tfidfer.extract_tags(split_whole, topK=10, withWeight=True, allowPOS=('nr', 'nz'))
        tag2w = tfidfer.extract_tags(split_whole, topK=10, withWeight=True,
                                     allowPOS=('an', 'b', 'j',  'l', 'Ng', 'n', 'nr', 'ns', 'nz', 'nt'))
        tag3w = textranker.textrank(split_whole, topK=10, withWeight=True,
                                    allowPOS=('an', 'b', 'j', 'k', 'l', 'Ng', 'n', 'nr', 'ns', 'nz', 'vn', 's'))
        #: 将所有的权重值提取出来形成一个list用于后面的随机选择
        # tag1 = []
        wei1 = []
        for tup in tag1w:
            # tag1.append(tup[0])
            wei1.append(tup[1])
        # tag2 = []
        wei2 = []
        for tup in tag2w:
            # tag2.append(tup[0])
            wei2.append(tup[1])
        # tag3 = []
        wei3 = []
        for tup in tag3w:
            # tag3.append(tup[0])
            wei3.append(tup[1])

        final_tagw = list()
        import random
        choose_size = 5     # 一个参数，控制随机选择时选择的数目
        if len(tag1w) < choose_size:
            final_tagw.extend(tag1w)
        else:
            cho1 = random.choices(tag1w, weights=wei1, k=choose_size)   # 根据分词器判断的权重加权随机选择choose_size个关键词
            final_tagw.extend(cho1)      # 合并每个提取器的提议

        if len(tag2w) < choose_size:
            final_tagw.extend(tag2w)
        else:
            cho2 = random.choices(tag2w, weights=wei2, k=choose_size)
            final_tagw.extend(cho2)

        if len(tag3w) < choose_size:
            final_tagw.extend(tag3w)
        else:
            cho3 = random.choices(tag3w, weights=wei3, k=choose_size)
            final_tagw.extend(cho3)

        final_tagw.sort(key=lambda x: float(x[1]), reverse=True)     # 所有提议的关键词按权重排序（有重复）
        tag = []
        wei = []
        for tup in final_tagw:
            tag.append(tup[0])
            wei.append(tup[1])

        final_tag = list()
        if len(tag) < 2 * choose_size:
            final_tag = tag
        else:
            choose = random.choices(tag, weights=wei, k=2*choose_size)
            final_tag = choose   # 再做一次按权重的随机抽取，大小为2*choose_size
        final_tag = set(final_tag + old_tag)      # 合并原标签并去重

        doc['tags'] = list(final_tag)   # 修改原标签
    json_file.close()

    with open(file, 'wt', encoding='utf-8') as fo:
        json.dump(doc_list, fo, ensure_ascii=False)
    # 打印出整个json，不写入文件


def tag_all_file(path_in: str):
    import os
    dealt_num = 0
    fail_tag = 0
    failed = 0
    for file in os.listdir(path_in):
        if file.endswith('.json'):
            file = os.path.abspath(file)  # 绝对路径
            try:
                tag_one_file2(file)
                dealt_num += 1
                logger.info('Done ' + str(dealt_num) + ', succeed tagging the file:' + file)
            except:
                logger.exception('Unpredefined error, here is the traceback:')    # taceback信息会被记录下来
                import json
                try:        # 尝试直接打开json并将内容打印
                    # json_file = open(file, 'r', encoding='utf-8')
                    # doc_list = json.load(json_file)
                    # print(doc_list)
                    fail_tag += 1
                    logger.info('Original ' + str(fail_tag) + ', fail tagging but use the original file:' + file)
                except IOError:
                    failed += 1
                    logger.exception('Failed ' + str(failed) + 'fail to open ' + file)  # 打开文件失败
                    continue
                except:
                    failed += 1
                    logger.exception('Failed ' + str(failed) + 'fail to load json file:' + file)  # 打开文件失败
                    continue
                continue  # 继续处理下一个文件
        else:
            pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='run tagging by cmd')
    parser.add_argument('--dir', type=str, default='.')
    args = parser.parse_args()      # 从命令行获取参数
    json_dir = args.dir
    tag_all_file(json_dir)


