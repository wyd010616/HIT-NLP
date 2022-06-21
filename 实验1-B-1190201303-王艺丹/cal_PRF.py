#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''将分词结果转换为区间集合，利用交集求TP'''
def trans_region(lines):
    total = []
    for line in lines:
        if line == '\n':
            continue
        region = []
        start = 0
        line = line[:len(line) - 1].split('/ ') # 去掉最后一个换行符，按'/'切分分词结果
        for w in line:
            end = start + len(w)# 计算每个词的区间
            region.append((start,end)) # 将该词的区间加入该行的list
            start = end # 更新区间起始坐标
        if len(region) > 0:
            total.append(region) # 将不为空的区间加入整个结果的list
    return total


# In[2]:


'''将seg&pos标准答案转换为区间'''
def trans_s(lines):
    total = []
    for line in lines:
        if line == '\n':
            continue
        region = []
        start = 0
        line = line[:len(line) - 1].split()
        for w in line:
            head, tail = w.split('/')
            if head[0] == '[':
                end = start + len(head) - 1
                region.append((start,end))
                start = end
                continue
            end = start + len(head)
            region.append((start,end))
            start = end
        if len(region) > 0:
            total.append(region)
    return total


# In[3]:


'''计算分词准确率，召回率，F值'''
def cal(an, mine):
    assert len(an) == len(mine)
    TP, P, R, size_A, size_B = 0, 0, 0, 0, 0
    for i in range(len(an)):
        A = set(an[i]) # 标准答案区间集合
        B = set(mine[i]) # 个人分词结果集合
        size_A += len(A)
        size_B += len(B)
        TP += len(A & B) # TP值即为正确划分的词语个数，即交集
    P = TP / size_B
    R = TP / size_A
    F = 2 * P * R / (P + R)
    return P, R, F


# In[4]:


'''计算seg_LM.txt的P，R，F并写入score.txt'''
def cal_prf():
    with open('./data/199801_seg&pos.txt', 'r', encoding= 'GBK') as s:
        part_test_s = trans_s(s.readlines())
    with open('./result/seg_LM.txt', 'r', encoding= 'utf-8') as lm:
        part_test_lm = trans_region(lm.readlines())
    p_lm, r_lm, f_lm = cal(part_test_s, part_test_lm)
    with open('./result/seg_LM_score.txt', 'w', encoding='utf-8') as score_txt: # 待写入分词结果文件
        score_txt.write("seg_LM.txt的准确率P为：" + '%F' %p_lm + '\n')
        score_txt.write("seg_LM.txt的召回率R为：" + '%F' %r_lm + '\n')
        score_txt.write("seg_LM.txt的F值F为：" + '%F' %f_lm + '\n')


# In[5]:


if __name__ == '__main__':
    cal_prf()

