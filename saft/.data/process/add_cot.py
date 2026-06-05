import json

with open("/volume/wzhang/ghchen/rzw/yy/.project/SaFT/saft/.data/numina-cot-10k.json", "r", encoding="utf-8") as f:
    data = json.load(f)
import pdb; pdb.set_trace()
print(type(data))
print(data)