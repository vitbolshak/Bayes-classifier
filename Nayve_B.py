# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:51:12 2019

@author: Admin
"""
from collections import defaultdict
from random import randrange
import numpy as np
import matplotlib.pyplot as plt

def flat_list(list_of_list):
    res = []
    for fold in list_of_list:
        for sample in fold:
            res.append(sample)
    return res
        
def train(samples):
    classes = defaultdict(lambda:0)
    class_num = defaultdict(lambda:0)
    tables = []
    features_count = len(samples[0][0])
    #заоминаем количество классов
    for line in samples:
        classes[line[1]] += 1
        
    #даёт классам числовой индекс
    k = 0
    all_keys = classes.keys()
    for cl in all_keys:
        class_num[cl] = k
        k += 1
        
    for i in range(features_count):     
        table = []  #таблица для i-го свойства
        for line in samples:
            value = line[0][i]
            label = line[1]
            flag = False

            for row in table:
                if value == row[0]:
                    flag = True
                    #если такое значение этого свойства уже встречалось 
                    #меняем его вероятность при такущем классе
                    row[1][class_num[label]] += 1 / classes[label]
                    break
            #если такое значение не встречаолсь, заводим новую строку с веротяностями    
            if not flag:
                a = np.zeros(len(classes))
                a[class_num[label]] += 1 / classes[label]
                row = [value]
                row.append(a)
                table.append(row)
        tables.append(table)

        
    for cl in classes:
        classes[cl] /= len(samples)
    return classes, tables, class_num


def predict(model, data):
    classes, likelyhood, class_num = model
    max = -1
    all_keys = classes.keys()
    
    for cl in all_keys:
        prob = 1
        prob *= classes[cl]
        
        for i in range(len(data)):
            table = likelyhood[i]
            for row in table:
                flag = False
                
                if row[0] == data[i]:
                    prob *= row[1][class_num[cl]]
                    flag = True
                    break
            #если такое значение i-го атрибута не было найдено         
            if not flag:
                prob *= 10**(-3)
        if prob > max:
            max = prob
            pred_class = cl
    return pred_class
    
def get_features(feats):
    features = []
    for feat in feats[:-1]:
        features.append(feat)
    
    return features

def get_label(feats):    
    return feats[-1]

path_to_data = 'C:\\Users\\Admin\\Kurva\\Michail\\'

def do_samples(path):
    file = open(path)
    samples = []
    for line in file:
        if line[-1] == '\n':
            line = line[:-1]
        feats = line.split(',')
        samples.append([get_features(feats),get_label(feats)])
    return samples

def split_samples(samples, k):
    copy = samples
    list_of_folds = []
    fold_len = int(len(samples)/k)

    for i in range(k):
        fold = []
        while len(fold) < fold_len:
            random_index = randrange(len(copy))
            fold.append(copy.pop(random_index))

        list_of_folds.append(fold)
    return list_of_folds

def run_cross_validation(samples,k):
    list_of_folds = split_samples(samples,k)
    acc = []
    counts = []
    for fold in list_of_folds:
        train_set = []
        for x in list_of_folds:
            train_set.append(x)
        test_set = []
        for x in fold:
            test_set.append(x)
        train_set.remove(test_set)  #всё ещё список спиков свойств              
        train_set = flat_list(train_set)
        
        #train_set обучающее множество fold - остается тестовым
        model = train(train_set)
        
        accuracy = 0
        for line in test_set:
            prediction = predict(model, line[0])
            if prediction == line[1]:
                accuracy += 1
        count = len(test_set)
        counts.append(count)
        acc.append(accuracy)
    return acc, counts
    
    

train_file = path_to_data + 'iris.data'   
samples = do_samples(train_file)
cross_validate = True
if cross_validate:
    k = 6
    acc, counts = run_cross_validation(samples, k)
    acc_count = []
    for i in range(len(acc)):
        acc_count.append(acc[i] / counts[i])
    print('Точность прогноза (в долях) при различных разбиениях :')
    print('№   кол-во верно предсказанных  кол-во эл-тов в тест. наборе  точность')
    for i in range(len(acc)):
        print(i, ':         ', acc[i], '                     ', counts[i], '                    ',acc_count[i])
        
    plt.plot(acc_count, color = 'r', label = 'точность прогноза', linestyle = '--')
    plt.ylabel('Точность')
    plt.xlabel('Номер разбиения')
    plt.legend()
else:
    
    model = train(samples)
    test_file = path_to_data + 'test.data'  
    op_file = open(test_file)
    count = 0
    acc = 0
    for line in op_file:    
        if line[-1] == '\n':
            line = line[:-1]
        line = line.split(',')
        print(get_features(line))
        prediction = predict(model, get_features(line))
        count += 1
        print('Предсказываю: ' , prediction)
        print('На самом деле: ', get_label(line))
        if prediction == get_label(line):
                acc += 1
    print('Колличество тестовых данных: ', count)
    acc = acc / count * 100
    print('Точность предсказаний: ', acc, '%')


            