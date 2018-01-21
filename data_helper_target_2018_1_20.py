"""
description: this file helps to load raw file and gennerate batch x  ------target file
author:许海明
date:11/1/2017
"""
import numpy as np


#file path
dataset_path='#######文件的路径##########'



def set_dataset_path(path):
    dataset_path=path


def read_file(filepath,vocabulary_size):
    f = open(filepath, 'r', encoding="UTF-8")  # 获得目标文件的指针
    sen_stop=vocabulary_size-2
    doc_stop=vocabulary_size-1
    doc_index=0                #记录已读取文章的个数
    max_target_sen_num=-1      #文章包含的最大的句子数
    max_target_word_num=-1     #句子包含的最大单词数
    train_target_set=[]
    length_array_eachdoc_target=[]            #记录每篇文章的每一句的单词的数 [ [该篇文章每一句的单词数目],[。。。] ]
    article=[]                         #正在读取的文章
    length_array_article=[]            #记录正在读取的文章的每一句的长度
    flag=0
    text_s = f.readline()
    while True:
        if not text_s:
            break  #判断是否到达文件的结尾

        text_s = text_s.strip().split()
        if len(text_s) == 0:             #一篇文章读取结束
            if len(article) == 0 or flag == 1:
                article = []
                length_array_article = []
                flag=0
                pass
            else:
                doc_index=doc_index+1
                article[-1].append(doc_stop)
                length_array_article[-1]+=1
                if len(article[-1])>max_target_word_num:
                    max_target_word_num=len(article[-1])

                if max_target_sen_num<len(article):
                    max_target_sen_num=len(article)
                train_target_set.append(article)
                length_array_eachdoc_target.append(length_array_article)
                article=[]
                length_array_article=[]
        else:
            text_s.append(sen_stop)
            if len(text_s) > 41:
                flag=1
                pass
            else:
                if max_target_word_num<len(text_s):
                    max_target_word_num=len(text_s)
                length_array_article.append(len(text_s))
                article.append(text_s)

        text_s = f.readline()


    num_doc=doc_index
    print("目标文件总共",num_doc,"篇文章")
    f.close()
    return (train_target_set,length_array_eachdoc_target,max_target_sen_num,max_target_word_num,num_doc)




def padding_and_generate_mask(train_target_set,new_train_target_set,mask_train_target_set,length_array_eachdoc_target,sen_mask,mask_train_target_set_float):
    for i,article in enumerate(train_target_set):
        for j,sent in enumerate(article):
            new_train_target_set[i][j,0:length_array_eachdoc_target[i][j]]=sent
            mask_train_target_set[j][0:length_array_eachdoc_target[i][j],i]=1
            mask_train_target_set_float[j][0:length_array_eachdoc_target[i][j],i]=1
            sen_mask[i][j]=1
    return new_train_target_set,mask_train_target_set,sen_mask,mask_train_target_set_float

def load_data(filepath,vocabulary_size):
    train_target_set, length_array_eachdoc_target, max_target_sen_num, max_target_word_num, num_doc=read_file(filepath,vocabulary_size)

    new_train_target_set=np.ones([num_doc,max_target_sen_num,max_target_word_num],dtype=np.int32) #生成的是矩阵
    mask_train_target_set=np.zeros([max_target_sen_num,max_target_word_num,num_doc],dtype=np.int32)
    mask_train_target_set_float=np.zeros([max_target_sen_num,max_target_word_num,num_doc],dtype=np.float32)
    sen_mask = np.zeros([num_doc, max_target_sen_num],dtype=np.int32)

    train_target_set,mask_train_target_set,sen_mask,mask_train_target_set_float=padding_and_generate_mask(train_target_set,new_train_target_set,mask_train_target_set,length_array_eachdoc_target,sen_mask,mask_train_target_set_float)  #(new_train_target_set,mask_train_target_set,sen_mask)
    data_target=(np.array(train_target_set),np.array(mask_train_target_set),length_array_eachdoc_target, max_target_sen_num, max_target_word_num, num_doc,np.array(sen_mask),np.array(mask_train_target_set_float))
    return data_target


def batch_iter(data_target,batch_size):
    train_target_set,mask_train_target_set, length_array_eachdoc_target, max_target_sen_num, max_target_word_num, num_doc,sen_mask,mask_train_target_set_float=data_target
    num_batches_per_epoch=int((num_doc-1)/batch_size)
    print("num_batches_per_epoch_target:",num_batches_per_epoch)
    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,num_doc)
        if end_index - start_index != batch_size:  # 说明最后一个batch的size小于batch_size采取舍弃的做法
            break
        else:
            return_train_target_set = train_target_set[start_index:end_index]
            return_mask_train_target_set = mask_train_target_set[:,:,start_index:end_index]
            return_mask_train_target_set_float = mask_train_target_set_float[:,:,start_index:end_index]
            yield (return_train_target_set,return_mask_train_target_set,return_mask_train_target_set_float)
