import sys
import re
import json

sys.stdout.reconfigure(encoding='utf-8')

# ------------ convert xml marks to format expected by the evaluator ----------------
def NER_eval_format(ex, text):
   xmlopen = "<drug>|<group>|<brand>|<drug_n>"

   original = text
   fmt = []
   p = re.search(xmlopen, text)
   while p is not None :
       os, oe = p.span()
       tag = p.group()[1:-1] # remove < >
       openlen = oe-os

       p = re.search(f"</{tag}>", text)  # find closing tag 

       ok = False
       if p is not None:
          cs, ce = p.span()       
          if f"<{tag}>" not in text[oe:cs]:
             # make sure closing tag is the closest
             drug = text[oe:cs]
             ok = True

       if not ok :
          print(f"Missing closing {tag} in: {original}", file=sys.stderr)

       text = text[:os]+text[oe:]  # remove opening mark
       if ok:
          # Closing tag indices were computed before removing the
          # opening tag, so shift them left by the opening tag length.
          cs -= openlen
          ce -= openlen
          text = text[:cs]+text[ce:]  # remove closing mark
          fmt.append(f"{ex['id']}|{os}-{cs-1}|{drug}|{tag}")

       p = re.search(xmlopen, text)

   return fmt       

#----------------------------

fname = sys.argv[1]

with open(fname) as jf:
    ds = json.load(jf)

for ex in ds:
    out = NER_eval_format(ex, ex["predicted"])
    if out : print("\n".join(out))
