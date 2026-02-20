
# Extracts ground truth entities or DDI in given data file, in the format
# expected by the evaluator.

# May be useful to compare with your output or to perform data exploration
import sys
import json
from xml.dom.minidom import parse

class DataFormatter() :

    def __init__(self, datafile) :
       self.tree =  parse(datafile)   
    
    def extract_NER(self, outfile) :
       dataset = []
       # process each sentence in the file
       sentences = self.tree.getElementsByTagName("sentence")
       for s in sentences :
          sid = s.attributes["id"].value   # get sentence id
          print(f"extracting sentence {sid}        \r", end="", file=sys.stderr)
          stext = s.attributes["text"].value.strip()   # get sentence text
          entities = s.getElementsByTagName("entity") # get gold standard entities
          spans = []
          for e in entities :
             # for discontinuous entities, we only get the first span
             # (will not work, but there are few of them)
             (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
             typ =  e.attributes["type"].value
             spans.append((int(start),int(end),typ)) 
          # start with the last span
          spans.sort(reverse=True)

          newtext = stext
          for s in spans :
              start,end,typ = s
              newtext = newtext[:end+1] + f"</{typ}>" + newtext[end+1:]
              newtext = newtext[:start] + f"<{typ}>" + newtext[start:]

          dataset.append({"id" : sid,
                          "input" : stext,
                          "output" : newtext
                         })
                     
       outf = open(outfile, "w") if type(outfile)==str else outfile
       json.dump(dataset, outf, indent=True, ensure_ascii=False)
       if type(outfile)==str : outf.close()
        
    def extract_DDI(self, outfile) :
       dataset = []
       # process each sentence in the file
       sentences = self.tree.getElementsByTagName("sentence")
       for s in sentences :
          sid = s.attributes["id"].value   # get sentence id
          print(f"extracting sentence {sid}        \r", end="", file=sys.stderr)
          stext = s.attributes["text"].value.strip()   # get sentence text
          entities = s.getElementsByTagName("entity") # get gold standard entities
          ents = {}
          for e in entities :
             # for discontinuous entities, we only get the first span
             # (will not work, but there are few of them)
             (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
             typ = e.attributes["type"].value
             eid = e.attributes["id"].value
             ents[eid] = {"type" : typ, "start" : int(start), "end" : int(end) }

          pairs = s.getElementsByTagName("pair")
          for p in pairs :
             pid = p.attributes["id"].value
             ddi = p.attributes["type"].value if p.attributes["ddi"].value=="true" else "none"
             e1 = ents[p.attributes["e1"].value]
             e2 = ents[p.attributes["e2"].value]
                     
             newtext = stext
             newtext = newtext[:e2["end"]+1] + f'</drug2>' + newtext[e2["end"]+1:]
             newtext = newtext[:e2["start"]] + f'<drug2>' + newtext[e2["start"]:]
             newtext = newtext[:e1["end"]+1] + f'</drug1>' + newtext[e1["end"]+1:]
             newtext = newtext[:e1["start"]] + f'<drug1>' + newtext[e1["start"]:]
   
             dataset.append({"id" : pid,
                             "input" : newtext,
                             "output" : ddi
                            })
                     
       outf = open(outfile, "w") if type(outfile)==str else outfile
       json.dump(dataset, outf, indent=True, ensure_ascii=False)
       if type(outfile)==str : outf.close()


if __name__ == "__main__" :
    task = sys.argv[1]
    datafile = sys.argv[2]
    
    fmt = DataFormatter(datafile)
    if task=="NER" :
        fmt.extract_NER(sys.stdout)
    elif task=="DDI" :
        fmt.extract_DDI(sys.stdout)
        
