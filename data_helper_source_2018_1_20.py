"""
description: this file helps to load raw file and gennerate batch x  ------source file
author:许海明
date:8/1/2017
"""
import numpy as np


'''
将整个文件读取出来
'''
def read_file(filepath):
    f = open(filepath, 'r', encoding="UTF-8")  # 获取源文件的文件指针
    doc_index=0                #记录已读取文章的个数
    max_source_sen_num=-1      #文章包含的最大的句子数
    max_source_word_num=-1     #句子包含的最大单词数
    train_source_set=[]
    length_array_eachdoc_source=[]            #记录每篇文章的每一句的单词的数 [ [该篇文章每一句的单词数目],[。。。] ]
    article=[]                         #正在读取的文章
    length_array_article=[]            #记录正在读取的文章的每一句的长度

    text_s = f.readline()
    flag=0                            #用于记录该篇文章中是否有大于指定大小的句子  有 舍弃
    while True:
        if not text_s:
            break  #判断是否到达文件的结尾

        text_s=text_s.strip().split()
        if len(text_s)==0:             #一篇文章读取结束
            if len(article)==0 or flag==1:
                flag=0
                article = []
                length_array_article = []
                pass
            else:
                doc_index=doc_index+1
                if max_source_sen_num<len(article):
                    max_source_sen_num=len(article)
                train_source_set.append(article)
                length_array_eachdoc_source.append(length_array_article)
                article=[]
                length_array_article=[]
        else:
            if len(text_s) > 40:
                flag = 1
                pass
            else:
                if max_source_word_num < len(text_s):
                    max_source_word_num = len(text_s)
                length_array_article.append(len(text_s))
                article.append(text_s)

        text_s = f.readline()

    num_doc=doc_index
    print("共",num_doc,"篇文章")
    f.close()
    return (train_source_set,length_array_eachdoc_source,max_source_sen_num,max_source_word_num,num_doc)





def padding_and_generate_mask(train_source_set,new_train_source_set,mask_train_source_set,length_array_eachdoc_source,sen_mask):
    for i,article in enumerate(train_source_set):
        for j,sent in enumerate(article):
            new_train_source_set[i][j,0:length_array_eachdoc_source[i][j]]=sent
            mask_train_source_set[j][0:length_array_eachdoc_source[i][j],i]=1
            sen_mask[i][j]=1
    return new_train_source_set,mask_train_source_set,sen_mask


def load_data(filepath):
    train_source_set, length_array_eachdoc_source, max_source_sen_num, max_source_word_num, num_doc=read_file(filepath)
    new_train_source_set=np.ones([num_doc,max_source_sen_num,max_source_word_num],dtype=np.int32) #生成的是矩阵
    mask_train_source_set=np.zeros([max_source_sen_num,max_source_word_num,num_doc],dtype=np.int32)
    sen_mask = np.zeros([num_doc, max_source_sen_num],dtype=np.int32)

    train_source_set,mask_train_source_set,sen_mask=padding_and_generate_mask(train_source_set,new_train_source_set,mask_train_source_set,length_array_eachdoc_source,sen_mask)  #(new_train_source_set,mask_train_source_set)

    data_source=(np.array(train_source_set),np.array(mask_train_source_set),length_array_eachdoc_source, max_source_sen_num, max_source_word_num,num_doc,np.array(sen_mask))
    return data_source


def batch_iter(data_source,batch_size):
    train_source_set, mask_train_source_set, length_array_eachdoc_source, max_source_sen_num, max_source_word_num, num_doc,sen_mask=data_source
    num_batches_per_epoch=int((num_doc-1)/batch_size)
    print("num_batches_per_epoch_source:",num_batches_per_epoch)
    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,num_doc)
        if end_index - start_index != batch_size:  # 说明最后一个batch的size小于batch_size采取舍弃的做法
            break
        else:
            return_sen_mask =sen_mask[start_index:end_index]
            return_train_source_set = train_source_set[start_index:end_index]
            return_mask_train_source_set = mask_train_source_set[:,:,start_index:end_index]
            yield (return_sen_mask,return_train_source_set,return_mask_train_source_set)
