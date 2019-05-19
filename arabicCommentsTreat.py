import emoji
import re
from googletrans import Translator
import normalize_dataset as nd
import sys






def normalize_emojis(str):
    translator = Translator()
    str = emoji.demojize(str)
    str = re.sub(r'[\:]+', ' ', str)
    str = re.sub(r'[_]+', ' ', str)
    str.strip()
    str=translator.translate(str,dest='ar').text
    return str


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

