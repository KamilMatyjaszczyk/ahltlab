import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import paths
from model_ep10 import FineTuning
from examples import Examples
from prompts import Prompts

# ------------ check command line and get arguments -----------------
def get_arguments():
    if not 6 <= len(sys.argv) <= 7 or (len(sys.argv) == 7 and sys.argv[6] != "-quant"):
        print(f"Usage: {sys.argv[0]} model prompts trainfile valfile strategy [-quant]", file=sys.stderr)
        print("strategy: balanced | random | null-heavy | no-int-heavy", file=sys.stderr)
        sys.exit(1)

    model = sys.argv[1]
    promptfile = sys.argv[2]
    traindata = sys.argv[3]
    valdata = sys.argv[4]
    strategy = sys.argv[5]
    quantized = (len(sys.argv) == 7)

    return model, promptfile, traindata, valdata, strategy, quantized



############## MAIN ################

# get command line arguments
model, promptfile, traindata, valdata, strategy, quantized = get_arguments()
print(f"========= FINE TUNE == MODEL={model}  quantized={quantized}", file=sys.stderr)

# load prompts
prompts = Prompts(promptfile)

# load model and tokenizer
t0 = time.time()
MODEL_PATH = f"/scratch/nas/1/PDI/mgl0/models/{model}"
engine = FineTuning(MODEL_PATH, quantized=quantized)
print(f"Model loading took {time.time()-t0:.1f} seconds", file=sys.stderr)

# load and tokenize datasets
t0 = time.time()
trainfile = os.path.join(paths.DATA, traindata + ".xml")
train_data = Examples(trainfile, "DDI")

if strategy == "balanced":
    train_examples = train_data.select_examples(5000, balanced=True)
elif strategy == "random":
    train_examples = train_data.select_examples(5000, balanced=False)
elif strategy == "null-heavy":
    train_examples = train_data.select_examples_null_heavy(5000)
elif strategy == "no-int-heavy":
    train_examples = train_data.select_examples_no_int_heavy(5000)
else:
    print(f"Unknown strategy: {strategy}", file=sys.stderr)
    sys.exit(1)

train_dataset = engine.tokenize_dataset(train_examples, prompts)

valfile = os.path.join(paths.DATA,valdata+".xml")
val_examples = Examples(valfile, "DDI").select_examples(500, balanced=True)
val_dataset = engine.tokenize_dataset(val_examples, prompts)
print(f"Dataset loading took {time.time()-t0:.1f} seconds", file=sys.stderr)
        
# Fine-tune the model and save results
t0 = time.time()
os.makedirs(paths.MODELS, exist_ok=True)
quant="-quant" if quantized else ""
outputdir = os.path.join(paths.MODELS, f"FT-{model}{quant}-{strategy}-ep10-lr1e5.weights")
print(f"Saving fine-tuned weights to {outputdir}", file=sys.stderr)
engine.train(train_dataset,
             val_dataset, 
             outputdir) 
print(f"Training took {time.time()-t0:.1f} seconds", file=sys.stderr)

print("Fine-tuning complete!", file=sys.stderr)

# clean up gpu
del engine
torch.cuda.empty_cache() 

