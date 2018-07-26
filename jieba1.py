# encoding=utf-8
import jieba as jb
import jieba.analyse as jbana
import logging
#: 分词
jb.setLogLevel(logging.ERROR)
seg_list = jb.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jb.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jb.cut("国安老板现身决赛 网友：要买U23姆巴佩[决赛闯入球场者身份查明 为俄反普京组织成员]"
                  "[贺炜:暴雨大比分令人难忘 魔笛是真正大师][马奎尔：世界杯后我还是我 会脚踏实地走下去]"
                  "[基恩：格列兹曼不该获点球 不怪VAR裁判太蠢]"
                  "[因凡蒂诺:VAR让足球变透明 判罚准确率近100%] [决赛VAR判罚点球遭质疑 基恩:判罚令我作呕]")  # 默认是精确模式
print(", ".join(seg_list))
seg_list = jb.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))


#: 自定义字典
# sentence1 = "国安老板现身决赛 网友：要买U23姆巴佩[决赛闯入球场者身份查明 为俄反普京组织成员][贺炜:暴雨大比分令人难忘 魔笛是真正大师][马奎尔：世界杯后我还是我 会脚踏实地走下去]"
# seg_list = jb.cut(sentence1)
# print(", ".join(seg_list))
# jb.add_word('姆巴佩')
# jb.add_word('马奎尔')
# seg_list = jb.cut(sentence1)
# print(", ".join(seg_list))


# #: 提取关键词
# doc1 = "每当瑞雪初霁，站在宝石山上向南眺望，西湖银装素裹，白堤横亘雪柳霜桃。断桥的石桥拱面无遮无拦，" \
#        "在阳光下冰雪消融，露出了斑驳的桥栏，而桥的两端还在皑皑白雪的覆盖下。依稀可辨的石桥身似隐似现，" \
#        "而涵洞中的白雪奕奕生光，桥面灰褐形成反差，远望去似断非断，故称断桥。伫立桥头，放眼四望，远山近水，尽收眼底，" \
#        "给人以生机勃勃的强烈属深刻的印象。"
# tags = jbana.extract_tags(doc1, topK=10, withWeight=True)
# print(tags)
#
# tags2 = jbana.textrank(doc1, topK=10, allowPOS=('ns', 'n'))
# print(",".join(tags2))

# with open('qssyr.txt', 'rt', encoding='utf-8') as txt:
#     doc2 = txt.read()
#     tags = jbana.extract_tags(doc2, topK=20)
#     print(",".join(tags))
#     tags2 = jbana.textrank(doc2, topK=20
#                            # , allowPOS=('ns', 'n', 'vn', 'v')
#                            , allowPOS=('m')
#                            #, allowPOS=('ns', 'nr', 'ns', 'nz', 'n', 'vn', 'an', 'Ng')
#     )
#     print(",".join(tags2))
#     # 与nlpir对比了一下，感觉提取效果稍差一点


# #: 词性标注
# import jieba_fast.posseg as pseg
# import jieba_fast as jb
# import logging
# jb.setLogLevel(log_level=logging.WARNING)
# cutter = pseg.POSTokenizer()
# doc1 = "每当瑞雪初霁，站在宝石山上向南眺望，西湖银装素裹，白堤横亘雪柳霜桃。断桥的石桥拱面无遮无拦，" \
#        "在阳光下冰雪消融，露出了斑驳的桥栏，而桥的两端还在皑皑白雪的覆盖下。依稀可辨的石桥身似隐似现，" \
#        "而涵洞中的白雪奕奕生光，桥面灰褐形成反差，远望去似断非断，故称断桥。伫立桥头，放眼四望，远山近水，尽收眼底，" \
#        "给人以生机勃勃的强烈属深刻的印象。"
# seg_list = jb.cut(doc1)
# print(", ".join(seg_list))
# print('-----------------')
# words = cutter.cut(doc1)
#
# for word, flag in words:
#     print('%s %s' % (word, flag))

