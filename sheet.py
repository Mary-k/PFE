'''
comment='الله أكبر الله أكبر الله أكبر الجهاد'

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

def wordRep(sent):
    empty=''
    empty=' '.join(unique_list(sent.split()))
    return empty

print(wordRep(comment))
'''
import emoji
import re
from googletrans import Translator

translator = Translator()



def normalize_emojis(str):
    str = emoji.demojize(str)
    str = re.sub(r'[\:]+', ' ', str)
    str = re.sub(r'[_]+', ' ', str)

    return str.strip()

sent='😱😊😄😉😆😋😁'
emoj=''
emoj=normalize_emojis(sent)
#print('lll'+emoj+'lll')

translation=translator.translate(emoj,dest='ar')
print(translation.text)
