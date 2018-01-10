import data_helper_source


f_source=open("data/debug",'r',encoding="UTF-8")
stop=0

while stop==0:
    train_source_set, mask_train_source_set, length_array_eachdoc_source, max_source_sen_num, max_source_word_num, batch_size,f_source,stop = data_helper_source.load_data(f_source, 16)
    print(mask_train_source_set.shape)
    if stop!=1:
        print(mask_train_source_set[0][0])