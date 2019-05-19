#!/usr/bin/python3

import string, sys, os

ar = open('arabicComments.txt', 'r')
totalAr = len(ar.readlines())
latin = open('latinComments.txt', 'r')
totalLatin = len(latin.readlines())
total =  totalAr + totalLatin

dictFr = open('french_comments.txt.dict', 'r')
dictFrCount = len(dictFr.readlines())
dictEn = open('english_comments.txt.dict', 'r')
dictEnCount = len(dictEn.readlines())
dictFrAr = open('french_arabise_comments.txt.dict', 'r')
dictFrArCount = len(dictFrAr.readlines())

# Print stats

print("Total number of entries: {} (100% of total entries)".format(total))

arRatio = (totalAr / total) * 100
print("Total number of Arabic entries: {} ({:.2f}% of total entries)".format(totalAr, arRatio))

latinRatio = (totalLatin / total) * 100
print("Total number of Latin entries: {} ({:.2f}% of total entries)".format(totalLatin, latinRatio))


print()
print("Among latin entries: ")

dictFrRatio = (dictFrCount / totalLatin) * 100
print("Total number of French entries: {} ({:.2f}% of total Latin entries)".format(dictFrCount, dictFrRatio))

dictEnRatio = (dictEnCount / totalLatin) * 100
print("Total number of English entries: {} ({:.2f}% of total Latin entries)".format(dictEnCount, dictEnRatio))


dictFrArRatio = (dictFrArCount / totalLatin) * 100
print("Total number of French-Arabise entries: {} ({:.2f}% of total Latin entries)".format(dictFrArCount, dictFrArRatio))