#! /usr/bin/python3

import sys, os
from xml.dom.minidom import parse
import spacy

from patterns import *

# Extra features:
FEATURE_SET = os.environ.get("FEATURE_SET", "baseline")

USE_CONTEXT_FEATURES = FEATURE_SET in {"context", "contextsyntax", "contextsyntaxentity", "all"}
USE_CLUE_VERB_FEATURES = FEATURE_SET in {"clue", "all"}
USE_EXTRA_SYNTAX_FEATURES = FEATURE_SET in {"syntax", "contextsyntax", "contextsyntaxentity", "all"}
USE_ENTITY_PATH_FEATURES = FEATURE_SET in {"entity_path", "contextsyntaxentity", "all"}

CLUE_VERBS = {
   # effect-related
   "increase", "decrease", "reduce", "enhance", "potentiate",
   "augment", "produce", "cause", "prolong",

   # mechanism-related
   "inhibit", "induce", "interfere", "antagonize",
   "alter", "affect", "impair", "delay", "displace",

   # advise-related
   "avoid", "recommend", "contraindicate", "consider",
   "require", "monitor", "warrant", "exceed",
   "titrate", "initiate", "prescribe", "discontinue",

   # interaction/administration-related
   "interact", "administer", "give", "coadministere"
}


## ------------------- 
## -- Convert a pair of drugs and their context in a feature vector

def add_context_features(feats, tree, entities, e1, e2):
   ent1 = entities[e1]
   ent2 = entities[e2]

   # Entity type combination
   feats.add("ctx_typePair=" + ent1["type"] + "_" + ent2["type"])

   # Entity order
   if ent1["start"] < ent2["start"]:
      feats.add("ctx_order=E1_before_E2")
      left_ent, right_ent = ent1, ent2
   else:
      feats.add("ctx_order=E2_before_E1")
      left_ent, right_ent = ent2, ent1

   # Character distance bucket
   char_dist = right_ent["start"] - left_ent["end"]

   if char_dist <= 10:
      feats.add("ctx_charDist=very_short")
   elif char_dist <= 30:
      feats.add("ctx_charDist=short")
   elif char_dist <= 80:
      feats.add("ctx_charDist=medium")
   else:
      feats.add("ctx_charDist=long")

   # Tokens between entities
   between_tokens = []
   for tk in tree:
      tk_start = tk.idx
      tk_end = tk.idx + len(tk.text)

      if left_ent["end"] < tk_start and tk_end < right_ent["start"]:
         between_tokens.append(tk)

   # Token distance bucket
   n_between = len(between_tokens)
   if n_between == 0:
      feats.add("ctx_tokensBetween=0")
   elif n_between <= 2:
      feats.add("ctx_tokensBetween=1-2")
   elif n_between <= 5:
      feats.add("ctx_tokensBetween=3-5")
   else:
      feats.add("ctx_tokensBetween=6+")

   # General POS information between entities
   if between_tokens:
      pos_seq = "_".join([tk.pos_ for tk in between_tokens[:5]])
      feats.add("ctx_betweenPOSSeq=" + pos_seq)

      has_verb = any(tk.pos_ == "VERB" for tk in between_tokens)
      has_noun = any(tk.pos_ == "NOUN" for tk in between_tokens)
      has_adj = any(tk.pos_ == "ADJ" for tk in between_tokens)
      has_adv = any(tk.pos_ == "ADV" for tk in between_tokens)

      if has_verb:
         feats.add("ctx_hasVerbBetween")
      if has_noun:
         feats.add("ctx_hasNounBetween")
      if has_adj:
         feats.add("ctx_hasAdjBetween")
      if has_adv:
         feats.add("ctx_hasAdvBetween")
   else:
      feats.add("ctx_noTokensBetween")

   # Third entity information
   third_between = False
   third_before = False
   third_after = False

   for eid, ent in entities.items():
      if eid in {e1, e2}:
         continue

      if left_ent["end"] < ent["start"] and ent["end"] < right_ent["start"]:
         third_between = True
         feats.add("ctx_thirdEntityTypeBetween=" + ent["type"])
      elif ent["end"] < left_ent["start"]:
         third_before = True
      elif right_ent["end"] < ent["start"]:
         third_after = True

   if third_between:
      feats.add("ctx_thirdEntityBetween")
   else:
      feats.add("ctx_noThirdEntityBetween")

   if third_before:
      feats.add("ctx_thirdEntityBefore")

   if third_after:
      feats.add("ctx_thirdEntityAfter")


def add_clue_verb_features(feats, tree, entities, e1, e2):
   ent1 = entities[e1]
   ent2 = entities[e2]

   if ent1["start"] < ent2["start"]:
      left_ent, right_ent = ent1, ent2
   else:
      left_ent, right_ent = ent2, ent1

   found_clue = False

   for tk in tree:
      lemma = tk.lemma_.lower()

      if lemma not in CLUE_VERBS:
         continue

      found_clue = True

      feats.add("clue_hasVerb")
      feats.add("clue_verb=" + lemma)

      tk_start = tk.idx
      tk_end = tk.idx + len(tk.text)

      if tk_end < left_ent["start"]:
         feats.add("clue_position=before")
         feats.add("clue_beforeVerb=" + lemma)

      elif left_ent["end"] < tk_start and tk_end <= right_ent["start"]:
         feats.add("clue_position=between")
         feats.add("clue_betweenVerb=" + lemma)

      elif right_ent["end"] < tk_start:
         feats.add("clue_position=after")
         feats.add("clue_afterVerb=" + lemma)

   if not found_clue:
      feats.add("clue_noVerb")

def add_extra_syntax_features(feats, tree, entities, e1, e2):
   ent1 = entities[e1]
   ent2 = entities[e2]

   # Find the syntactic head token of each entity
   tkE1 = get_fragment_head(tree, ent1["start"], ent1["end"])
   tkE2 = get_fragment_head(tree, ent2["start"], ent2["end"])

   if tkE1 is None or tkE2 is None:
      return

   # Find the lowest common subsumer of the entity heads
   lcs = get_LCS(tree, tkE1, tkE2)

   if lcs is None:
      return

   # -------------------------------------------------
   # 1. Generalised dependency path length
   # -------------------------------------------------
   def distance_to_ancestor(node, ancestor):
      """Return the number of dependency edges from node to ancestor."""
      if node == ancestor:
         return 0

      distance = 0
      current = node

      while current != ancestor and current.head != current:
         current = current.head
         distance += 1

      if current == ancestor:
         return distance

      return None

   dist1 = distance_to_ancestor(tkE1, lcs)
   dist2 = distance_to_ancestor(tkE2, lcs)

   if dist1 is not None and dist2 is not None:
      path_length = dist1 + dist2

      if path_length <= 2:
         feats.add("syn_pathLength=short")
      elif path_length <= 5:
         feats.add("syn_pathLength=medium")
      else:
         feats.add("syn_pathLength=long")

   # Get dependency-path nodes for the coordination features below
   path1 = get_up_path(tkE1, lcs)
   path2 = get_down_path(lcs, tkE2)

   if path1 is not None and path2 is not None:

      # -------------------------------------------------
      # 2. Coordination information on the dependency path
      # -------------------------------------------------
      path_nodes = path1 + path2
      conj_count = sum(1 for node in path_nodes if node.dep_ == "conj")

      if conj_count >= 1:
         feats.add("syn_coordinationPath")

      if conj_count >= 2:
         feats.add("syn_manyConjunctions")

   # -------------------------------------------------
   # 3. Generalised verb-role pattern
   # -------------------------------------------------
   # Find the first governing verb at or above the LCS
   governing_verb = lcs

   while governing_verb.pos_ != "VERB" and governing_verb.head != governing_verb:
      governing_verb = governing_verb.head

   if governing_verb.pos_ == "VERB":
      verb_path1 = get_up_path(tkE1, governing_verb)
      verb_path2 = get_up_path(tkE2, governing_verb)

      if verb_path1 is not None and verb_path2 is not None:
         if len(verb_path1) > 0 and len(verb_path2) > 0:
            role1 = verb_path1[-1].dep_
            role2 = verb_path2[-1].dep_

            feats.add("syn_generalVerbRoles=" + role1 + "_" + role2)

def add_entity_path_features(feats, tree, entities, e1, e2):
   ent1 = entities[e1]
   ent2 = entities[e2]

   # Find the syntactic head token of each target entity
   tkE1 = get_fragment_head(tree, ent1["start"], ent1["end"])
   tkE2 = get_fragment_head(tree, ent2["start"], ent2["end"])

   if tkE1 is None or tkE2 is None:
      return

   lcs = get_LCS(tree, tkE1, tkE2)

   if lcs is None:
      return

   path1 = get_up_path(tkE1, lcs)
   path2 = get_down_path(lcs, tkE2)

   if path1 is None or path2 is None:
      return

   # Full path, including the LCS once
   full_path = list(path1)

   if lcs not in full_path:
      full_path.append(lcs)

   for node in path2:
      if node not in full_path:
         full_path.append(node)

   # -------------------------------------------------
   # 1. Identify non-target entities occurring on the path
   # -------------------------------------------------
   third_entities_on_path = set()

   for tk in full_path:
      for eid, ent in entities.items():
         if eid in {e1, e2}:
            continue

         tk_start = tk.idx
         tk_end = tk.idx + len(tk.text)

         if ent["start"] <= tk_start and tk_end <= ent["end"] + 1:
            third_entities_on_path.add(eid)

   if third_entities_on_path:
      feats.add("epath_thirdEntityOnPath")

      for eid in third_entities_on_path:
         feats.add("epath_thirdEntityTypeOnPath=" + entities[eid]["type"])

      if len(third_entities_on_path) >= 2:
         feats.add("epath_multipleThirdEntitiesOnPath")
   else:
      feats.add("epath_noThirdEntityOnPath")

   # -------------------------------------------------
   # 2. Entity-masked dependency path
   # -------------------------------------------------
   def node_representation(tk):
      # Replace any entity token by ENTITY, keeping its dependency role
      for eid, ent in entities.items():
         tk_start = tk.idx
         tk_end = tk.idx + len(tk.text)

         if ent["start"] <= tk_start and tk_end <= ent["end"] + 1:
            return "ENTITY_" + tk.dep_

      # Preserve governing verbs, since they may describe the interaction
      if tk.pos_ == "VERB":
         return tk.lemma_.lower() + "_" + tk.dep_

      # For other nodes, preserve only the dependency relation
      return tk.dep_

   path_length = len(path1) + len(path2)

   # Only store detailed masked paths for short/medium paths.
   # Long paths are often highly specific list structures.
   if path_length <= 5:
      masked_path1 = "<".join(node_representation(tk) for tk in path1)
      masked_lcs = node_representation(lcs)
      masked_path2 = ">".join(node_representation(tk) for tk in path2)

      feats.add(
         "epath_masked="
         + masked_path1
         + "<"
         + masked_lcs
         + ">"
         + masked_path2
      )
   else:
      feats.add("epath_maskedPathTooLong")

def extract_pair_features(tree, entities, e1, e2) :
   feats = set()

   # Features about entity types
   feats.add("typeE1="+ entities[e1]['type'])
   feats.add("typeE2="+ entities[e2]['type'])
   if entities[e1]['text'].lower() == entities[e2]['text'].lower() : 
      feats.add("samedrug")

   if USE_CONTEXT_FEATURES:
      add_context_features(feats, tree, entities, e1, e2)

   if USE_CLUE_VERB_FEATURES:
      add_clue_verb_features(feats, tree, entities, e1, e2)
   
   if USE_EXTRA_SYNTAX_FEATURES:
      add_extra_syntax_features(feats, tree, entities, e1, e2)

   if USE_ENTITY_PATH_FEATURES:
      add_entity_path_features(feats, tree, entities, e1, e2)
      
   # features about paths in the tree.
   # get head token for each gold entity
   tkE1 = get_fragment_head(tree,entities[e1]['start'],entities[e1]['end'])
   tkE2 = get_fragment_head(tree,entities[e2]['start'],entities[e2]['end'])
   if tkE1 is not None and tkE2 is not None:      
      # get LCS      
      lcs = get_LCS(tree,tkE1,tkE2)

      if lcs is not None :
          feats.add("lcs="+lcs.lemma_+"_"+lcs.pos_)
          
          # paths from E1 to LCS, using lemma, rel, or both
          path1 = get_up_path(tkE1,lcs)
          p1 = "<".join([x.lemma_+"_"+x.dep_ for x in path1])
          feats.add("path1="+p1)
          p1b = "<".join([x.lemma_ for x in path1])
          feats.add("path1b="+p1b)
          p1c = "<".join([x.dep_ for x in path1])
          feats.add("path1c="+p1c)

          # paths from LCS to E2, using lemma, rel, or both
          path2 = get_down_path(lcs,tkE2)
          p2 = ">".join([x.lemma_+"_"+x.dep_ for x in path2])
          feats.add("path2="+p2)
          p2b = ">".join([x.lemma_ for x in path2])
          feats.add("path2b="+p2b)
          p2c = ">".join([x.dep_ for x in path2])
          feats.add("path2c="+p2c)

          # paths from E1 to E2, using lemma, rel, or both
          p = p1+"<"+lcs.lemma_+"_"+lcs.dep_+">"+p2
          feats.add("path="+p)
          pb = p1b+"<"+lcs.lemma_+">"+p2b
          feats.add("pathb="+pb)
          pc = p1c+"<"+lcs.dep_+">"+p2c
          feats.add("pathc="+pc)

          # LCS lemma/tag and rels under it
          if len(path1)>0 and len(path2)>0 :
             pa = path1[-1].dep_+"<"+lcs.lemma_+">"+path2[0].dep_
             feats.add("pathA="+pa)
             pab = path1[-1].dep_+"<"+lcs.pos_+">"+path2[0].dep_
             feats.add("pathAb="+pab)

          # words in path from E1 to E2
          for w in path1 :
             feats.add("wip1="+w.lemma_)
             feats.add("wip="+w.lemma_)
          for w in path2 :
             feats.add("wip2="+w.lemma_)
             feats.add("wip="+w.lemma_)
          feats.add("wip="+lcs.lemma_)
          feats.add("lcs="+lcs.lemma_)

          # lcs children
          for w in lcs.children : feats.add("lcsCH="+w.lemma_)
      
   # features using rule-based patterns
   for pat in patterns :
      match = patterns[pat](tree, entities, e1, e2)
      if match is not None: 
         for m in match :
            feats.add(pat+"="+m)
                     
   return feats


## --------- Feature extractor ----------- 
## -- Extract features for each entity pair in each
## -- sentence in given file

def extract_features(datafile, outfile, dump_trees=False) :

   # open output file
   outf = open(outfile, "w")
   if dump_trees:
       treedir = os.path.join(os.path.dirname(outfile), "svg")
       os.makedirs(treedir, exist_ok=True)
    
   # create spacy parser
   nlp = spacy.load("en_core_web_trf",
                    enable=["transformer", "tagger","attribute_ruler", "lemmatizer", "ner", "parser"])

   # parse XML file, obtaining a DOM tree
   tree = parse(datafile)

   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        print(f"extracting sentence {sid}             \r", end="")
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")           
           text = e.attributes["text"].value
           typ = e.attributes["type"].value
           entities[id] = {'start': int(offs[0]), 'end': int(offs[-1]),
                           'text': text, 'type' : typ}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1 : continue

        # get syntactic analysis for the sentence
        analysis = nlp(stext)
        if dump_trees : 
           svg = spacy.displacy.render(analysis,style="dep")    
           with open(os.path.join(treedir,sid+".svg"),"w") as sf :  
              sf.write(svg)       
        
        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi=="true") : dditype = p.attributes["type"].value
            else : dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction
            feats = extract_pair_features(analysis,entities,id_e1,id_e2) 
            # resulting vector
            print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t", file=outf)


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir outfile
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- corresponding feature vectors to outfile
## --

if __name__ == "__main__" :
    # directory with files to process
    datafile = sys.argv[1]
    # file where to store results
    featfile = sys.argv[2]
    trees = len(sys.argv)>3 and sys.argv[3]=="trees"
    
    extract_features(datafile, featfile, trees)

