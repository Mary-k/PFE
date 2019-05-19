#!/usr/bin/python3

import string, sys, os, re
import pyarabic.araby as araby

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

def cleanLine(line):
    line = re.sub(r'http\S+', '', line) # URLs
    line = re.sub(r'\#[0-9a-zA-Z\_]*', '', line) # Hashtags
    line = re.sub(r'[0-9]{1,3}\:[0-9]{1,3}', '', line) # Time-links (like 15:45)
    line = normalize_arabic(line) # Fix arabic things
    return line.rstrip("\r\n")


def detectCodeSwitch(line):
    latinLetters = re.findall(r'[a-zA-Z]', line)
    arabicLetters = re.findall(r'[ا-ي]', line)

    return len(latinLetters) > 0 and len(arabicLetters) > 0

if len(sys.argv) < 2:
    print("Please provide an input source as the first argument")
    os._exit(1)

if not os.path.isfile(sys.argv[1]):
    print("Input file does not exist.")
    os._exit(1)

file = open(sys.argv[1], "r")
data = file.readlines()
dataLength = len(data)
codeSwitchedLines = 0

for line in data:
    line = cleanLine(line)
    if detectCodeSwitch(line):
        codeSwitchedLines = codeSwitchedLines + 1

codeSwitchedRatio = (codeSwitchedLines / dataLength) * 100
print("Total number of Code Switched entries: {} ({:.2f}% of total entries)".format(codeSwitchedLines, codeSwitchedRatio))

