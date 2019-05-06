from sklearn.feature_extraction.text import CountVectorizer
import csv
'''
dataset_list=[]
temp=[]
with open('/home/mimi/Desktop/PFE/DATASETS/youtube_comments_fixed.csv', 'r') as f:
    reader = csv.reader(f)
    #dataset_list=
    dataset_list=list(reader)
    
print(dataset_list[10])

print(dataset_list)
print(len(dataset_list))
'''
'''
with open('/home/mimi/Desktop/PFE/DATASETS/algerianYoutubeDataset.txt', 'r') as f:
    dataset_list = [line.strip() for line in f]
print(len(dataset_list))

vect = CountVectorizer()

vect.fit(dataset_list)

print("Vocabulary size: {}".format(len(vect.vocabulary_)))
#print("Vocabulary content:\n {}".format(vect.vocabulary_))

import pyarabic.araby as araby
import pyarabic.number as number

from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()
'''
dataset_list=[]
with open('/home/mimi/Desktop/PFE/DATASETS/algerianYoutubeDataset.txt', 'r') as f:
    dataset_list = [line.strip() for line in f]

print(len(dataset_list))

from alphabet_detector import AlphabetDetector
ad = AlphabetDetector()

latinList=[]
arabicList=[]
for comment in dataset_list:    
    if ad.only_alphabet_chars(comment, "LATIN"):
        latinList.append(comment)
    else:
        arabicList.append(comment)
'''

with open('/home/mimi/Desktop/PFE/DATASETS/latinComments.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(latinList))

with open('/home/mimi/Desktop/PFE/DATASETS/arabicComments.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(arabicList))

print(len(arabicList))

print(len(latinList))'''

