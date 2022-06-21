#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy


# In[2]:


class WordMatching:
    def __init__(self, dic_path = './result/dic.txt'):
        self.tree, self.MAX_LEN = self.__get_info__(dic_path)
        
    def __get_info__(self, path):
        dict_list, MAX_LEN = [], 0
        with open(path, 'r', encoding= 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dict_list.append(line.split())
                MAX_LEN = max(MAX_LEN,len(line))
        return dict_list, MAX_LEN
    
    def FMM(self, test_path, result_path = './result/seg_FMM.txt'):
        with open(test_path, 'r', encoding='GBK') as txt: # 打开待分词文件
            lines = txt.readlines()
        with open(result_path, 'w', encoding='utf-8') as seg_FMM: # 待写入分词结果文件
            for sentence in lines:
                if sentence == '\n':
                    seg_FMM.write(sentence) # 写入段落换行符
                    continue
                div_result = [sentence[:19]]
                sentence = sentence[19:].strip()
                while len(sentence) > 0:
                    div = sentence[0 : min(len(sentence),self.MAX_LEN)] # 最长字符匹配
                    while div not in dict_list and len(div) > 1: # 不在词典内且不为单字
                        div = div[0:len(div)-1] # 删去最后一个字，继续匹配查找词典
                    div_result.append(div) # 将分词结果写入
                    sentence = sentence[len(div):] # 删去匹配成功的词，继续匹配
                seg_FMM.write('/ '.join(div_result) + '\n') # 处理完成一行写入文件
                
    def BMM(self, test_path, result_path = './result/seg_BMM.txt'):
        with open(test_path, 'r', encoding='GBK') as txt: # 打开待分词文件  使用with as减少代码量
            lines = txt.readlines()
        with open(result_path, 'w', encoding='utf-8') as seg_BMM: # 待写入分词结果文件
            for sentence in lines:
                div_result = []
                if sentence == '\n':
                    seg_BMM.write(sentence) # 写入段落换行符
                    continue
                date = sentence[:19]
                sentence = sentence[19:].strip()
                while len(sentence) > 0:
                    div = sentence if len(sentence)<= self.MAX_LEN else sentence[len(sentence) - self.MAX_LEN :] # 最长字符匹配
                    while div not in dict_list and len(div) > 1: # 不在词典内且不为单字
                        div = div[1:] # 删去第一个字，继续匹配查找词典
                    div_result.insert(0, div)  # 将分词结果暂存，逆向匹配，分词结果倒序
                    sentence = sentence[:len(sentence)-len(div)]# 删去匹配成功的词，继续匹配
                div_result.insert(0, date)
                seg_BMM.write('/ '.join(div_result) + '\n') # 处理完成一行写入文件


# In[3]:


if __name__ == '__main__':
    mm = WordMatching()
    mm.FMM('./data/199801_sent.txt')
    mm.BMM('./data/199801_sent.txt')

