#! /usr/bin/python3

import sys, os
import re
from xml.dom.minidom import parse
import spacy

import paths
from dictionaries import Dictionaries

## --------- get tag -----------
##  Find out whether given token is marked as part of an entity in the XML
def get_label(tks, tke, spans):
    for (spanS, spanE, spanT) in spans:
        if tks == spanS and tke <= spanE + 1:
            return "B-" + spanT
        elif tks > spanS and tke <= spanE + 1:
            return "I-" + spanT
    return "O"


def get_shape(token):
    shape = []
    for ch in token:
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        else:
            shape.append(ch)
    return "".join(shape)


def get_compact_shape(token):
    s = get_shape(token)
    if not s:
        return s

    compact = [s[0]]
    for ch in s[1:]:
        if ch != compact[-1]:
            compact.append(ch)
    return "".join(compact)


def extract_sentence_features(tokens, dicts):

    # for each token, generate list of features and add it to the result
    sentenceFeatures = {}

    for i, tk in enumerate(tokens):
        tokenFeatures = []
        t = tk.text

        # -------------------------
        # Current token features
        # -------------------------
        tokenFeatures.append("form=" + t)
        tokenFeatures.append("formlower=" + t.lower())

        if len(t) >= 2:
            tokenFeatures.append("suf2=" + t[-2:])
        if len(t) >= 3:
            tokenFeatures.append("suf3=" + t[-3:])
        if len(t) >= 4:
            tokenFeatures.append("suf4=" + t[-4:])
        if len(t) >= 5:
            tokenFeatures.append("suf5=" + t[-5:])

        if len(t) >= 3:
            tokenFeatures.append("pref3=" + t[:3])
        if len(t) >= 4:
            tokenFeatures.append("pref4=" + t[:4])

        tokenFeatures.append("shape=" + get_shape(t))
        tokenFeatures.append("shapecompact=" + get_compact_shape(t))

        L = len(t)
        if L == 1:
            tokenFeatures.append("len=1")
        elif L == 2:
            tokenFeatures.append("len=2")
        elif L == 3:
            tokenFeatures.append("len=3")
        elif L == 4:
            tokenFeatures.append("len=4")
        elif L == 5:
            tokenFeatures.append("len=5")
        else:
            tokenFeatures.append("len=6+")

        if t.isupper():
            tokenFeatures.append("isUpper")
        if t.istitle():
            tokenFeatures.append("isTitle")
        if t.isdigit():
            tokenFeatures.append("isDigit")
        if t.islower():
            tokenFeatures.append("isLower")
        if t.isalpha():
            tokenFeatures.append("isAlpha")
        if t.isalnum():
            tokenFeatures.append("isAlnum")

        if "-" in t:
            tokenFeatures.append("hasDash")
        if "/" in t:
            tokenFeatures.append("hasSlash")
        if "(" in t or ")" in t:
            tokenFeatures.append("hasParen")
        if "," in t:
            tokenFeatures.append("hasComma")
        if "." in t:
            tokenFeatures.append("hasPeriod")

        if any(ch.islower() for ch in t):
            tokenFeatures.append("hasLower")
        if any(ch.isupper() for ch in t):
            tokenFeatures.append("hasUpper")
        if re.search("[0-9]", t):
            tokenFeatures.append("hasDigit")

        if len(t) > 0 and t[0].isupper():
            tokenFeatures.append("startsUpper")
        if len(t) > 0 and t[0].islower():
            tokenFeatures.append("startsLower")

        if any(ch.islower() for ch in t) and any(ch.isupper() for ch in t):
            tokenFeatures.append("mixedCase")

        found, val = dicts.find(t.lower(), "external")
        if found:
            for c in val:
                tokenFeatures.append("external=" + c)

        found, val = dicts.find(t.lower(), "externalpart")
        if found:
            for c in val:
                tokenFeatures.append("externalpart=" + c)

        # -------------------------
        # Previous token features
        # -------------------------
        if i > 0:
            tPrev = tokens[i - 1].text
            tokenFeatures.append("formPrev=" + tPrev)
            tokenFeatures.append("formlowerPrev=" + tPrev.lower())

            if len(tPrev) >= 2:
                tokenFeatures.append("suf2Prev=" + tPrev[-2:])
            if len(tPrev) >= 3:
                tokenFeatures.append("suf3Prev=" + tPrev[-3:])
            if len(tPrev) >= 4:
                tokenFeatures.append("suf4Prev=" + tPrev[-4:])
            if len(tPrev) >= 5:
                tokenFeatures.append("suf5Prev=" + tPrev[-5:])

            if len(tPrev) >= 3:
                tokenFeatures.append("pref3Prev=" + tPrev[:3])
            if len(tPrev) >= 4:
                tokenFeatures.append("pref4Prev=" + tPrev[:4])

            tokenFeatures.append("shapePrev=" + get_shape(tPrev))
            tokenFeatures.append("shapecompactPrev=" + get_compact_shape(tPrev))

            L = len(tPrev)
            if L == 1:
                tokenFeatures.append("lenPrev=1")
            elif L == 2:
                tokenFeatures.append("lenPrev=2")
            elif L == 3:
                tokenFeatures.append("lenPrev=3")
            elif L == 4:
                tokenFeatures.append("lenPrev=4")
            elif L == 5:
                tokenFeatures.append("lenPrev=5")
            else:
                tokenFeatures.append("lenPrev=6+")

            if tPrev.isupper():
                tokenFeatures.append("isUpperPrev")
            if tPrev.istitle():
                tokenFeatures.append("isTitlePrev")
            if tPrev.isdigit():
                tokenFeatures.append("isDigitPrev")
            if tPrev.islower():
                tokenFeatures.append("isLowerPrev")
            if tPrev.isalpha():
                tokenFeatures.append("isAlphaPrev")
            if tPrev.isalnum():
                tokenFeatures.append("isAlnumPrev")

            if "-" in tPrev:
                tokenFeatures.append("hasDashPrev")
            if "/" in tPrev:
                tokenFeatures.append("hasSlashPrev")
            if "(" in tPrev or ")" in tPrev:
                tokenFeatures.append("hasParenPrev")
            if "," in tPrev:
                tokenFeatures.append("hasCommaPrev")
            if "." in tPrev:
                tokenFeatures.append("hasPeriodPrev")

            if any(ch.islower() for ch in tPrev):
                tokenFeatures.append("hasLowerPrev")
            if any(ch.isupper() for ch in tPrev):
                tokenFeatures.append("hasUpperPrev")
            if re.search("[0-9]", tPrev):
                tokenFeatures.append("hasDigitPrev")

            if len(tPrev) > 0 and tPrev[0].isupper():
                tokenFeatures.append("startsUpperPrev")
            if len(tPrev) > 0 and tPrev[0].islower():
                tokenFeatures.append("startsLowerPrev")

            if any(ch.islower() for ch in tPrev) and any(ch.isupper() for ch in tPrev):
                tokenFeatures.append("mixedCasePrev")

            found, val = dicts.find(tPrev.lower(), "external")
            if found:
                for c in val:
                    tokenFeatures.append("externalPrev=" + c)

            found, val = dicts.find(tPrev.lower(), "externalpart")
            if found:
                for c in val:
                    tokenFeatures.append("externalpartPrev=" + c)
        else:
            tokenFeatures.append("BoS")

        # -------------------------
        # Next token features
        # -------------------------
        if i < len(tokens) - 1:
            tNext = tokens[i + 1].text
            tokenFeatures.append("formNext=" + tNext)
            tokenFeatures.append("formlowerNext=" + tNext.lower())

            if len(tNext) >= 2:
                tokenFeatures.append("suf2Next=" + tNext[-2:])
            if len(tNext) >= 3:
                tokenFeatures.append("suf3Next=" + tNext[-3:])
            if len(tNext) >= 4:
                tokenFeatures.append("suf4Next=" + tNext[-4:])
            if len(tNext) >= 5:
                tokenFeatures.append("suf5Next=" + tNext[-5:])

            if len(tNext) >= 3:
                tokenFeatures.append("pref3Next=" + tNext[:3])
            if len(tNext) >= 4:
                tokenFeatures.append("pref4Next=" + tNext[:4])

            tokenFeatures.append("shapeNext=" + get_shape(tNext))
            tokenFeatures.append("shapecompactNext=" + get_compact_shape(tNext))

            L = len(tNext)
            if L == 1:
                tokenFeatures.append("lenNext=1")
            elif L == 2:
                tokenFeatures.append("lenNext=2")
            elif L == 3:
                tokenFeatures.append("lenNext=3")
            elif L == 4:
                tokenFeatures.append("lenNext=4")
            elif L == 5:
                tokenFeatures.append("lenNext=5")
            else:
                tokenFeatures.append("lenNext=6+")

            if tNext.isupper():
                tokenFeatures.append("isUpperNext")
            if tNext.istitle():
                tokenFeatures.append("isTitleNext")
            if tNext.isdigit():
                tokenFeatures.append("isDigitNext")
            if tNext.islower():
                tokenFeatures.append("isLowerNext")
            if tNext.isalpha():
                tokenFeatures.append("isAlphaNext")
            if tNext.isalnum():
                tokenFeatures.append("isAlnumNext")

            if "-" in tNext:
                tokenFeatures.append("hasDashNext")
            if "/" in tNext:
                tokenFeatures.append("hasSlashNext")
            if "(" in tNext or ")" in tNext:
                tokenFeatures.append("hasParenNext")
            if "," in tNext:
                tokenFeatures.append("hasCommaNext")
            if "." in tNext:
                tokenFeatures.append("hasPeriodNext")

            if any(ch.islower() for ch in tNext):
                tokenFeatures.append("hasLowerNext")
            if any(ch.isupper() for ch in tNext):
                tokenFeatures.append("hasUpperNext")
            if re.search("[0-9]", tNext):
                tokenFeatures.append("hasDigitNext")

            if len(tNext) > 0 and tNext[0].isupper():
                tokenFeatures.append("startsUpperNext")
            if len(tNext) > 0 and tNext[0].islower():
                tokenFeatures.append("startsLowerNext")

            if any(ch.islower() for ch in tNext) and any(ch.isupper() for ch in tNext):
                tokenFeatures.append("mixedCaseNext")

            found, val = dicts.find(tNext.lower(), "external")
            if found:
                for c in val:
                    tokenFeatures.append("externalNext=" + c)

            found, val = dicts.find(tNext.lower(), "externalpart")
            if found:
                for c in val:
                    tokenFeatures.append("externalpartNext=" + c)
        else:
            tokenFeatures.append("EoS")

        sentenceFeatures[i] = tokenFeatures

    return sentenceFeatures


## --------- Feature extractor -----------
## -- Extract features for each token in each
## -- sentence in each file of given dir

def extract_features(datafile, outfile):

    # load dictionaries
    dicts = Dictionaries(os.path.join(paths.RESOURCES, "dictionaries.json"))

    # open output file
    outf = open(outfile, "w")

    # create analyzer. We don't need the parser now, it will be faster if disabled
    nlp = spacy.load("en_core_web_trf", enable=["tokenizer"])

    # parse XML file, obtaining a DOM tree
    tree = parse(datafile)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value   # get sentence id
        print(f"extracting sentence {sid}        \r", end="")
        spans = []
        stext = s.attributes["text"].value   # get sentence text
        entities = s.getElementsByTagName("entity")   # get gold standard entities

        for e in entities:
            # for discontinuous entities, we only get the first span
            # (will not work, but there are few of them)
            (start, end) = e.attributes["charOffset"].value.split(";")[0].split("-")
            typ = e.attributes["type"].value
            spans.append((int(start), int(end), typ))

        # convert the sentence to a list of tokens
        tokens = nlp(stext)

        # extract sentence features
        features = extract_sentence_features(tokens, dicts)

        # print features in format expected by CRF/SVM/MEM trainers
        for i, tk in enumerate(tokens):
            # see if the token is part of an entity
            tks, tke = tk.idx, tk.idx + len(tk.text)

            # get gold standard tag for this token
            tag = get_label(tks, tke, spans)

            # print feature vector for this token
            print(sid, tk.text, tks, tke - 1, tag, "\t".join(features[i]), sep="\t", file=outf)

        # blank line to separate sentences
        print(file=outf)

    # close output file
    outf.close()


## --------- MAIN PROGRAM -----------
## --
## -- Usage: baseline-NER.py target-dir outfile
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- corresponding feature vectors to outfile
## --

if __name__ == "__main__":
    # directory with files to process
    datafile = sys.argv[1]

    # file where to store results
    featfile = sys.argv[2]

    extract_features(datafile, featfile)