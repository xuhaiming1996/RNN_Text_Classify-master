"""
description: this file helps to load raw file and gennerate batch x  ------source file
author:许海明
date:8/1/2017
"""
import numpy as np


#file path
dataset_path='#######文件的路径##########'

def set_dataset_path(path):
    dataset_path=path


def read_file(f,batch_size):
    '''
    按照给定的batch_size读取数据，
    返回数据集train_source_set,实际的读取的batch_size,max_source_sen_nums,max_source_word_nums,stop
    '''
    doc_index=0                #记录已读取文章的个数
    max_source_sen_num=-1      #文章包含的最大的句子数
    max_source_word_num=-1     #句子包含的最大单词数
    train_source_set=[]
    length_array_eachdoc_source=[]            #记录每篇文章的每一句的单词的数 [ [该篇文章每一句的单词数目],[。。。] ]
    stop=0                             #是否达到文件结尾 0：没有 1：文件结束
    article=[]                         #正在读取的文章
    length_array_article=[]            #记录正在读取的文章的每一句的长度

    text_s = f.readline()
    while True:
        if not text_s:
            stop=1
            break  #判断是否到达文件的结尾

        text_s=text_s.strip().split()
        if len(text_s)==0:             #一篇文章读取结束
            if len(article)==0:
                pass
            else:
                doc_index=doc_index+1
                if max_source_sen_num<len(article):
                    max_source_sen_num=len(article)
                train_source_set.append(article)
                length_array_eachdoc_source.append(length_array_article)
                article=[]
                length_array_article=[]
                if doc_index==batch_size:
                    break
        else:
            if max_source_word_num<len(text_s):
                max_source_word_num=len(text_s)
            length_array_article.append(len(text_s))
            article.append(text_s)

        text_s = f.readline()


    batch_size=doc_index

    return (train_source_set,length_array_eachdoc_source,max_source_sen_num,max_source_word_num,batch_size,f,stop)



def padding_and_generate_mask(train_source_set,new_train_source_set,mask_train_source_set,length_array_eachdoc_source,sen_mask):
    for i,article in enumerate(train_source_set):
        for j,sent in enumerate(article):
            new_train_source_set[i][j,0:length_array_eachdoc_source[i][j]]=sent
            mask_train_source_set[j][0:length_array_eachdoc_source[i][j],i]=1
            sen_mask[i][j]=1
    return new_train_source_set,mask_train_source_set,sen_mask


def load_data(f,batch_size):
    train_source_set, length_array_eachdoc_source, max_source_sen_num, max_source_word_num, batch_size, f,stop=read_file(f,batch_size)
    new_train_source_set=np.ones([batch_size,max_source_sen_num,max_source_word_num],dtype=np.int32) #生成的是矩阵
    mask_train_source_set=np.zeros([max_source_sen_num,max_source_word_num,batch_size],dtype=np.int32)
    sen_mask = np.zeros([batch_size, max_source_sen_num],dtype=np.int32)

    train_source_set,mask_train_source_set,sen_mask=padding_and_generate_mask(train_source_set,new_train_source_set,mask_train_source_set,length_array_eachdoc_source,sen_mask)  #(new_train_source_set,mask_train_source_set)


    return (np.array(train_source_set),np.array(mask_train_source_set),length_array_eachdoc_source, max_source_sen_num, max_source_word_num,  batch_size, f ,stop, np.array(sen_mask))