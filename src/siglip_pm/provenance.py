"""Record exact source, geometry, model and environment without touching prior results."""
import hashlib,json,sys,subprocess
from pathlib import Path
import numpy as np,torch,transformers,timm
from src.siglip_pm.model import MODEL_PATH,MODEL_ID,REVISION

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
run=Path(sys.argv[1]);paths=list(Path('src/siglip_pm').glob('*.py'))+[Path('src/hcf/model.py'),Path('src/readout/models.py'),Path('src/pacoattr/official_eval.py')]
paths += list((run/'shared').glob('*boxes.npy'))+[run/'shared/image_ids.npy']
paths += [Path(MODEL_PATH)/n for n in ['config.json','preprocessor_config.json','model.safetensors']]
r=dict(backbone=MODEL_ID,revision=REVISION,python=sys.version,torch=torch.__version__,transformers=transformers.__version__,timm=timm.__version__,numpy=np.__version__,
       git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths})
json.dump(r,open(run/'provenance.json','w'),indent=1)
print('Provenance recorded:',len(paths),'files',flush=True)
