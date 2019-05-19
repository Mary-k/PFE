import normalize_dataset as nd
import sys
import os
import re

def remove_repetitions(str):
    # Remove the last 
    #str = str.rstrip("\n\r")

    # Remove latin duplicates:
    # What this does is, it captures the first caracter,
    # Then it replaces every consecutive duplicates of that caracter 
    # starting from 2 with the same caracter.
    # So for example: 
    # The word "Hello" stays the same.
    # The word "Maaaaaan" becomes "Man"
    # The word "Helloooooo" becomes "Hello"
    # This might not be the optimal solution, and it is quite naive.
    # But it works, and it doesn't change the meaning of words
    # (At least not from the perspective of sentiment analysis)
    str = re.sub(r'([a-zA-Z])\1{2,}', r'\1', str)
    str = re.sub(r'\s*[0-9]+\s+', ' ', str)
    str = re.sub(r'[ ]{2,}', ' ', str)
    str = re.sub(r'^[ ]', '', str)
    return str

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

def wordRep(sent):
    empty=''
    empty=' '.join(unique_list(sent.split()))
    return empty


if len(sys.argv) < 3:
    print("Please profide an input and output file.")
    print("The first argument is the input, the second is the output.")
    os._exit(1)




# Start processing file
file = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w+")

for line in file:
    line = nd.remove_punct(line)

    # This needs to after remove_punct & normalize_arabic
    line = nd.remove_repetitions(line)
    line = nd.replace_abbrv(line)
    line = nd.normalize_emojis(line)
    line=wordRep(line)
    if len(line) > 0:
        outFile.write(line+'\n')
   
    
