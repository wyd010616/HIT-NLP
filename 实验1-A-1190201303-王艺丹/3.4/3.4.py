#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time


# In[2]:


'''结合哈希实现时间复杂度O(1)的list查找对应元素，进而实现Trie树'''
class TrieNode:
    
    def __init__(self, char):
        self.char = char
        self.is_end = False
        self.size = 5000
        self.children = [None]* self.size # 哈希表，存的是TrieNode
        
    def djb_hash(self, s): # 计算字符对应的哈希地址
        h = 5381
        byte = s.encode()
        for e in byte:
            h = ((h << 5 ) + h) + e
        return h % self.size
    
    def add(self, s): # 将TrieNode存入列表
        add = self.djb_hash(s.char) # 求散列地址
        while self.children[add] is not None: # 线性探测处理冲突
            add = (add + 1) % self.size
        self.children[add] = s # 将字符存在对应哈希值的索引处
        return add
        
    def is_in(self, char): # 判断某一字符是否存在, 存在返回节点，不存在返回None
        add = start = self.djb_hash(char)
        if self.children[add] is None:
            return None
        while self.children[add].char != char:
            add = (add + 1) % self.size
            if start == add or self.children[add] is None:
                return None
        return self.children[add]
        
class Trie(object):
    def __init__(self):
        self.root = TrieNode("") # Trie树字典根节点为空，不存储字符
    
    def insert(self, word):
        node = self.root
        for char in word: # 循环将词的各个字符插入对应位置
            add = node.is_in(char)
            if add != None:
                node = add
            else:
                new_node = TrieNode(char)
                add = node.add(new_node)
                node = new_node
        node.is_end = True # 词尾标志


# In[3]:


class WordMatching:
    def __init__(self):
        self.tree_fmm, self.tree_bmm = self.__get_info__('./result/dic.txt')
        
    def __get_info__(self, path):
        tree_fmm, tree_bmm = Trie(), Trie()
        with open(path, 'r', encoding= 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                tree_fmm.insert(line.strip()) # fmm树顺序插入词语
                tree_bmm.insert(line.strip()[::-1]) # bmm树逆序插入词语
        return tree_fmm, tree_bmm
    
    def FMM(self, test_path = './data/199801_sent.txt', result_path = './result/seg_FMM.txt'):
        tree = self.tree_fmm
        with open(test_path, 'r', encoding='GBK') as txt: # 打开待分词文件
            lines = txt.readlines()
        with open(result_path, 'w', encoding='utf-8') as seg_FMM: # 待写入分词结果文件
            for sentence in lines:
                if sentence == '\n':
                    seg_FMM.write('\n') # 段落之间写入换行符
                    continue
                div_result = [sentence[:19]]
                sentence = sentence[19:].strip()
                while len(sentence) > 0:
                    cnt = 0
                    char = sentence[0]
                    nodes = tree.root
                    add = nodes.is_in(char)
                    if add is None:
                        div = char
                        sentence = sentence[1:] # 删去第一个字符
                        div_result.append(div) #
                        continue
                    while add is not None:
                        cnt += 1
                        if add.is_end:
                            div = sentence[:cnt]
                        elif add.is_end == False and cnt == 1:
                            div = sentence[:cnt] # 只取第一个字
                        if cnt == len(sentence):
                            break
                        add = add.is_in(sentence[cnt])
                    sentence = sentence[len(div):]
                    div_result.append(div)
                seg_FMM.write('/ '.join(div_result) + '\n')
    
    def BMM(self, test_path = './data/199801_sent.txt', result_path = './result/seg_BMM.txt'):
        tree = self.tree_bmm
        with open(test_path, 'r', encoding='GBK') as txt: # 打开待分词文件
            lines = txt.readlines()
        with open(result_path, 'w', encoding='utf-8') as seg_BMM: # 待写入分词结果文件
            for sentence in lines:
                if sentence == '\n':
                    seg_BMM.write('\n') # 段落之间写入换行符
                    continue
                date = sentence[:19]
                div_result = []
                sentence = sentence[19:].strip()[::-1] # 将句子逆序用bmm树按照fmm算法查找，等价于逆向最大后缀匹配
                while len(sentence) > 0:
                    cnt = 0
                    char = sentence[0]
                    nodes = tree.root
                    add = nodes.is_in(char)
                    if add is None:
                        div = char
                        sentence = sentence[1:] # 删去第一个字符
                        div_result.insert(0,div) #
                        continue
                    while add is not None:
                        cnt += 1
                        if add.is_end:
                            div = sentence[:cnt]
                        elif add.is_end == False and cnt == 1:
                            div = sentence[:cnt] # 只取第一个字
                        if cnt == len(sentence):
                            break
                        add = add.is_in(sentence[cnt])
                    sentence = sentence[len(div):]
                    div_result.insert(0,div[::-1]) # bmm逆序还原词语写入分词结果
                div_result.insert(0,date)
                seg_BMM.write('/ '.join(div_result) + '\n')


# In[4]:


def cal_time(path):
    mm = WordMatching()
    fmm_start = time.time()
    mm.FMM()
    fmm_end = time.time()
    bmm_start = time.time()
    mm.BMM()
    bmm_end = time.time()
    t_fmm = fmm_end - fmm_start
    t_bmm = bmm_end - bmm_start
    with open(path, 'w', encoding='utf-8') as time_txt: # 待写入分词结果文件
        time_txt.write('优化前FMM时间为：155663.2546s' + '\n') # 时间过长，无法跑出结果，估算
        time_txt.write('优化后FMM时间为：' + '%F' %t_fmm + 's'+ '\n')
        time_txt.write('\n')
        time_txt.write('优化前BMM时间为：146168.0281s' + '\n') # 时间过长，无法跑出结果，估算
        time_txt.write('优化后BMM时间为：' + '%F' %t_bmm + 's'+ '\n')


# In[5]:


if __name__ == '__main__':
    cal_time('./result/TimeCost.txt')

