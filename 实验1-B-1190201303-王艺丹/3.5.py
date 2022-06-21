#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from math import log


# In[2]:


'''判断有无日期序号，并返回结束位置'''
def str_match(sent):
    reg = '[0-9-/a-z]+'
    loc = re.match(reg, sent)
    if loc is not None:
        return len(loc.group())
    return 0


# In[3]:


'''将19801与19802数据整合用于训练，构建离线二元语法词典'''
def get_dic(dic_path = './data/train_all.txt', result_path = './result/bi_dic.txt'):
    with open(dic_path, 'r', encoding='GBK') as f: # 打开待处理文件  使用with as减少代码量
        lines = f.readlines()
    with open(result_path, 'w', encoding='utf-8') as dic: # 待写入词典文件
        bi_dic = {'EOS':{}}
        for line in lines:
            if line == '\n':
                continue
            line = line.split()[1:] # 去掉日期序号
            line.insert(0, 'BOS/ ')
            for i in range(1,len(line)):
                pre = line[i-1][1 if line[i-1][0] == '[' else 0 : line[i-1].index('/')]
                word = line[i][1 if line[i][0] == '[' else 0 : line[i].index('/')]
                if i == len(line) - 1: # 处理最后一个单词
                    bi_dic['EOS'][word] = bi_dic['EOS'].get(word, 0) + 1 
                else:
                    if word not in bi_dic:
                        bi_dic[word] = {}
                    bi_dic[word][pre] = bi_dic[word].get(pre, 0) + 1
        bi_dic = {key : bi_dic[key] for key in sorted(bi_dic.keys())} # 排序
        for word, word_list in bi_dic.items():
            for pre, freq in word_list.items():
                dic.write(word + ' ' + pre + ' ' + str(freq) + '\n')
    return bi_dic


# In[4]:


'''整合'''
class Bi_gram:
    def __init__(self, dic_path = './result/bi_dic.txt'):
        self.bi_dic, self.freq_dic = self.get_bi_freq_dics(dic_path) # 从路径读入离线二元语法词典构成bi_dic和词频字典freq_dic
        self.words_num = len(self.freq_dic) # 词典的词数目
        self.prefix_dic = self.get_prefix_dic() # 通过频率词典freq_dic获得前缀词典prefix_dic

    '''读入txt字典，获得bi_dic, freq_dic'''
    def get_bi_freq_dics(self, dic_path): 
        bi_dic = {}
        freq_dic = {}
        with open(dic_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, pre, freq = line.strip().split() # word，pre，P(word|pre)
            if word in bi_dic:
                bi_dic[word][pre] = int(freq)
                freq_dic[word] += int(freq)
            else:
                bi_dic[word] = {}
                bi_dic[word][pre] = int(freq)
                freq_dic[word] = int(freq)
        del freq_dic['EOS']
        with open('./result/freq_dic.txt', 'w', encoding = 'utf-8') as d:
            for word, freq in freq_dic.items():
                d.write(word + ' ' + str(freq) + '\n')
        return bi_dic, freq_dic
    
    '''根据词频字典构建前缀字典'''
    def get_prefix_dic(self):
        prefix_dic = {}
        for word in self.freq_dic:
            for i in range(len(word)):
                w = word[0:len(word) - i]
                if w not in prefix_dic:
                    prefix_dic[w] = self.freq_dic.get(w, 0)
        with open('./result/prefix_dic.txt', 'w', encoding = 'utf-8') as d:
            for word, freq in prefix_dic.items():
                d.write(word + ' ' + str(freq) + '\n')
        return prefix_dic
    
    '''根据前缀词典及句子，构建DAG词图'''
    def get_DAG(self, sent):
        DAG = {}
        N = len(sent)
        for k in range(N):
            tmplist = []
            i = k
            w = sent[k]
            while i < N and w in self.prefix_dic:
                if self.prefix_dic[w] > 0: # 说明统计词典中有词sent[k:i]，将该词加入k为key对应的列表中
                    tmplist.append(i)
                i += 1
                w = sent[k:i + 1] # 更新前缀
            if not tmplist: # 如果为单字，tmplist为空，将k本身加入
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG
    
    '''计算负对数概率'''
    def cal_logp(self, pre, word):
        pre_freq = self.freq_dic.get(pre, 0)  # 计算前词词频
        if word not in self.bi_dic:
            logp = log(self.words_num)
        else:
            word_freq = self.bi_dic[word].get(pre, 0) # P(word|pre)
            logp = - log(word_freq + 1 ) + log(pre_freq + self.words_num) # 拉普拉斯平滑处理
        return logp
    
    '''根据DAG生成带权有向图'''
    def get_gragh(self, sent, DAG):
        graph = {}
        ll = len(DAG)
        for i in range(ll + 1):
            if i == 0: # BOS
                tmplist = {}
                for j in DAG[i]:
                    w = sent[i:j+1]
                    logp = self.cal_logp('BOS', w) # P(w|'BOS')
                    tmplist[(i,j+1)] = logp
                graph['BOS'] = tmplist
            else:
                for j in DAG[i-1]:
                    tmplist = {}
                    pre = sent[i-1: j+1]
                    pre_r = (i - 1,j + 1)
                    if j == ll - 1: # 词尾
                        logp = self.cal_logp(pre, 'EOS') # P(to|w)
                        tmplist['EOS'] = logp
                    else:
                        for k in DAG[j + 1]:  
                            to = sent[j + 1: k + 1]
                            logp = self.cal_logp(pre, to) # P(to|pre)
                            tmplist[(j+1,k+1)] = logp
                    graph[pre_r] = tmplist
        return graph

    '''逆序遍历有向图，获得最短路径'''
    def veterbi(self, graph):
        pre_list = {} # 维护每个节点的最短路径上的前驱节点
        pre_list["BOS"] = (None, 0)
        sent = '在这一年中'
        for pre, v in graph.items():
            for to, weight in v.items():
                if pre not in pre_list:
                    continue
                cur = pre_list[pre][1] + weight
                if to not in pre_list:
                    pre_list[to] = (pre, cur)
                else:
                    if cur < pre_list[to][1]:
                        pre_list[to] = (pre, cur)
        # 逆向遍历获得路径
        prob = pre_list['EOS'][1]
        key = pre_list['EOS'][0]
        route = ['EOS']
        while key != 'BOS':
            route.insert(0, key)
            key = pre_list[key][0]
        return prob, route
    
    '''分词，返回概率与分词结果列表'''
    def seg_sent(self, sent):
        loc = str_match(sent)
        div_result = []
        if loc != 0: #判断包不包含日期序列
            div_result = [sent[:loc]]
            sent = sent[loc:]
        dag = self.get_DAG(sent) # 根据句子及词典生成DAG
        gra = self.get_gragh(sent, dag) # 根据DAG生成带权有向图
        prob, route = self.veterbi(gra) # 逆向获得最短路径
        for re in route:
            if re == 'EOS': # 到达句尾
                break
            word = sent[re[0]:re[1]] # 对应切分出的词语
            div_result.append(word)
        return prob, div_result


# In[5]:


def seg_LM(sent_path = './data/199801_sent.txt', result_path = './result/seg_LM.txt'):
    mm = Bi_gram(dic_path='./result/bi_dic.txt') # 读入离线utf-8编码的词典，用于初始化二元文法模型
    with open(sent_path, 'r', encoding='GBK') as sent:
        lines = sent.readlines()
    with open(result_path, 'w', encoding='utf-8') as seg:
        for sent in lines:
            if sent == '\n':
                seg.write('\n')
                continue
            prob, div_result = mm.seg_sent(sent.strip())
            seg.write('/ '.join(div_result) + '\n')


# In[6]:


# 获得离线字典
bi_dic = get_dic()


# In[ ]:


if __name__ == 'main':
    seg_LM('./data/199802_sent.txt', result_path = './result/seg_LM_19802.txt')
    seg_LM('./data/199801_sent.txt', result_path = './result/seg_LM.txt') # 改为对应的测试集路径即可，GBK编码

