#
# This script normalizes the dataset by doing the following:
# 1. Removing punctuations
# 2. Translating emojis to their textual representations
# 3. It normalizes arabic: Removes hamza caracters, strip tatweel & tachkeel
# 4. It removes concecutive repetitive caracters
# 5. It replaces abbreviations with normal words.
#
import sys
import emoji
import os
import string
import pyarabic.araby as araby
import re

# Just remove punctuation caracters.
def remove_punct(str):
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations

    str = str.translate(str.maketrans('', '', punctuations_list))
    return str

def normalize_arabic(str):
    # Some normalisations
    str = re.sub("[إأآا]", "ا", str)
    str = re.sub("ى", "ي", str)
    str = re.sub("ؤ", "ء", str)
    str = re.sub("ئ", "ء", str)
    str = re.sub("ة", "ه", str)
    str = re.sub("گ", "ك", str)
    str = araby.normalize_hamza(str)

    # Replace arabic comma by a space
    str = re.sub(r'[،]', ' ', str)

    # Remove arabic tatwil caracter
    str = re.sub(r'[ـ]+', '', str)
    str = araby.strip_tatweel(str)
    
    # Remove arabic tachkil
    str = araby.strip_tashkeel(str)
    str = araby.strip_lastharaka(str)

    return str
    
def remove_repetitive_numbers(str):
    str = re.sub(r'\s*[0-9]+\s+', ' ', str)
    str = re.sub(r'[ ]{2,}', ' ', str)
    str = re.sub(r'^[ ]', '', str)
    return str
    
def remove_repetitions(str):
    # Remove the last 
    str = str.rstrip("\n\r")

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

    # Now to arabic duplicates
    # Since arabic does not accept two concecutive letters as valid arabic.
    # The solution is quite simple, 
    # we just remove the duplicates instead of replacing it.
    str = re.sub(r'([ا-ي])\1{1,}', r'\1', str)
    return str


# Replace algerian abbreviations to french words
def replace_abbrv(str):

    # The key is the regex matching the abbreviation
    # The value is the replacement.
    abbrevMapping = {
        "([b]+[z]+[a]*[f]+)|(bcp[s]*)": "beaucoups",
        "b[e]*b[e]*": "bébé",
        "b[o]*[n]*[8]+": "goodnight",
        "p[r]*k[o]*[i]*": "pourquoi",
        "jms": "jamais",
        "tjr[s]*": "toujours",
        "d[z|s]+l": "desole",
        "d[a]*[c|k]+[r]*": "daccord",
        "m[e]*rc[i|6]*": "merci",
        "g[a]+[a|3]+": "tous",
        "d[e]*r[1]+": "derien",
        "b[1]+": "bien",
        "m[a|3]+a": "avec",
        "t[e]*la": "t'est la",
        "tel": "telephone",
        "c[v]+": "ca va",
        "m[e]*s[a]*g": "message",
        "p[e]*rs[o]*n": "personne",
        "j[c]+": "je sais",
        "j[e]*[s]+": "je suis",
        "q[e]*lq[1]+": "quelqu un",
        "p[r]*s[e]*k": "parce que",
        "[i|e]+[h]+": "oui",
        "r[a]*bi": "dieu",
        "m[a|e]*z[a|e]*l": "",
        "[l]*h[a]*md[o]*[l|i|a|hi|u]*": "merci dieu",
        "j[o|u]*rs": "jours",
        "tkt": "t'inquiete",
        "r[e]*p[o|n]*d[s]*": "reponds",
        "nchlh": "si dieu le veut",
        "p[a]*r[t]+": "partout",
        "ch[a]*q": "chaque",
        "w[e|a]*[l]+[e|a]*h": "je te jure",
        "pr" :"pour",
        "g[e]*nr[e]*": "genre",
        "b[e]*[r|k]+": "",
        "b[e|a]*[s]+[e|a]*h": "mais",
        "l[e|a]*z[e|a]*m": "il faut",
        "n[n]+": "non",
        "g[r]+[a]*[v]+": "grave",
        "j[e]*t[e]*m": "je t'aime",
        "b[i]*[s|x|z]+": "bisous",
        "w[i]+": "oui",
    }

    for abrv, rep in abbrevMapping.items():
        reg = re.compile("{0}".format(abrv), re.IGNORECASE)
        str = reg.sub(rep, str)
    
    return str

# Replace emojis with their textual representation
# Also remove any punctuations associated (like _ or : )
def normalize_emojis(str):
    str = emoji.demojize(str)
    str = re.sub(r'[\:]+', ' ', str)
    #str = re.sub(r'[_]+', ' ', str)
    
    return str
'''

if len(sys.argv) < 3:
    print("Please profide an input and output file.")
    print("The first argument is the input, the second is the output.")
    os._exit(1)

# Start processing file
file = open(sys.argv[1], "r")
outFile = open(sys.argv[2], "w+")

for line in file:
    line = remove_punct(line)
    line = normalize_arabic(line)

    # This needs to after remove_punct & normalize_arabic
    line = remove_repetitions(line)
    line = replace_abbrv(line)
    line = normalize_emojis(line)

    # Write line to output file
    outFile.write(line)
'''