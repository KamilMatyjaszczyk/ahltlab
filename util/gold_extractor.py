
# Extracts ground truth entities or DDI in given data file, in the format
# expected by the evaluator.

# May be useful to compare with your output or to perform data exploration
import sys
from xml.dom.minidom import parse

class GoldExtractor() :

    def __init__(self, datafile) :
       self.tree =  parse(datafile)   
    
    def extract_NER(self, outfile) :
       if type(outfile)==str : outf = open(outfile, "w") 
       else : outf = outfile
       
       entities = self.tree.getElementsByTagName("entity")
       for e in entities :
          sent_id = ".".join(e.attributes["id"].value.split(".")[:-1])
          print(sent_id,
                e.attributes["charOffset"].value,
                e.attributes["text"].value,
                e.attributes["type"].value,
                sep="|",
                file = outf)
       
       if type(outfile)==str : outf.close()
        
    def extract_DDI(self, outfile) :
       if type(outfile)==str : outf = open(outfile, "w") 
       else : outf = outfile
       pairs = self.tree.getElementsByTagName("pair")
       for p in pairs :
           if (p.attributes["ddi"].value=="true") :
               print(p.attributes["e1"].value,
                     p.attributes["e2"].value,
                     p.attributes["type"].value,
                     sep="|",
                     file = outf)
       
       if type(outfile)==str : outf.close()


if __name__ == "__main__" :
    task = sys.argv[1]
    datafile = sys.argv[2]
    
    gold = GoldExtractor(datafile)
    if task=="NER" :
        gold.extract_NER(sys.stdout)
    elif task=="DDI" :
        gold.extract_DDI(sys.stdout)
