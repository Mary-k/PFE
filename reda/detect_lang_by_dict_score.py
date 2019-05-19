#!/usr/bin/python
import sys, os, re, string

def remove_punct(str):
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations

    str = str.translate(str.maketrans('', '', punctuations_list))
    return str

def remove_repetitions(str):
    str = str.rstrip("\n\r")
    str = re.sub(r'([a-zA-Z])\1{2,}', r'\1', str)

def replace_abbrv(str):
    abbrevMapping = {
        "([b]+[z]+[a]*[f]+)|(bcp[s]*)": "beaucoups",
        "b[e]*b[e]*": "bebe",
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
        "vid": "video",
    }

    for abrv, rep in abbrevMapping.items():
        reg = re.compile("{0}".format(abrv), re.IGNORECASE)
        str = reg.sub(rep, str)
    return str


def isNeutralDominante(words, neutral, maxScore):
    if neutral['score'] == 0 or maxScore['lang'] == 'neutral':
        return maxScore

    neutralWordsRatioMustBeOver = 0.6
    neutralWordsRatio = neutral['score'] / len(words)
    print(words, maxScore['lang'], neutralWordsRatio)

    if maxScore['lang'] == 'fr' and neutralWordsRatio >= neutralWordsRatioMustBeOver:
        return neutral

    if maxScore['lang'] == 'en' and neutralWordsRatio >= neutralWordsRatioMustBeOver:
        return neutral

    return maxScore

def normalize_word(word):
    word = re.sub(r"[#]*", '', word)
    return word.lower()

def getMaxScore(score):
    return score["score"]

# French words dict
frFile = open('french_words.txt', 'r')
frWords = set(word.rstrip("\n\r") for word in frFile.readlines())
frFile.close()

# English words dict
enFile = open('english_words.txt', 'r')
enWords = set(word.rstrip("\n\r") for word in enFile.readlines())
enFile.close()

if len(sys.argv) < 2:
    print("Please profide an input file. (letinComments.txt)")
    os._exit(1)

# Start processing file
file = open(sys.argv[1], "r")

frFile = open('french_comments.txt.dict', 'w+')
enFile = open('english_comments.txt.dict', 'w+')
neutralFile = open('french_arabise_comments.txt.dict', 'w+')

for line in file:
    line = remove_punct(line)
    line = replace_abbrv(line)
    words = line.split()
    frScore = 0
    enScore = 0
    neutralScore = 0

    for word in words:
        word = normalize_word(word)
        if word in frWords:
            frScore = frScore + 1
        elif word in enWords:
            enScore = enScore + 1
        else:
            neutralScore = neutralScore + 1

    # Group scores by language
    scores = [
        { "lang":"fr", "score":frScore },
        { "lang":"en", "score":enScore },
        { "lang":"neutral", "score":neutralScore  }
    ]

    # Get the max score
    maxScore = max(scores, key=getMaxScore)
    maxScore = isNeutralDominante(words, scores[2], maxScore)
    ll = maxScore["lang"]

    if ll == 'fr':
        frFile.write(line)
    elif ll == 'en':
        enFile.write(line)
    elif ll == 'neutral':
        neutralFile.write(line)
    else:
        print("Man could not know what this is!")





