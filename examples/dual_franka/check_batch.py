"""GPU acceptance test using an actual converted episode; no robot interfaces."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from gr00t.configs.base_config import Config
from gr00t.model import MODEL_REGISTRY
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag,MessageType
from examples.dual_franka.run import read_config
from examples.dual_franka.modality import CONFIG
raw=read_config('examples/dual_franka/train_smoke.local.yaml');raw.pop('mode');raw.pop('deployment')
c=Config().load_dict(raw);c.data.modality_configs={'new_embodiment':CONFIG}
out=Path('deployment_records/batch_check');out.mkdir(parents=True,exist_ok=True)
p=MODEL_REGISTRY.get(type(c.model))(c,out);p.setup()
model=p.return_model().cuda();processor=p.return_processor();processor.eval()
loader=LeRobotEpisodeLoader(c.data.datasets[0].dataset_paths[0],CONFIG,decoder_kwargs={'num_ffmpeg_threads':1})
s=extract_step_data(loader[0],0,CONFIG,EmbodimentTag.NEW_EMBODIMENT)
msg=[{'type':MessageType.EPISODE_STEP.value,'content':s}]
batch=processor.collator([processor(msg)])
assert tuple(batch['inputs']['action_mask'].shape)==(1,40,132)
assert batch['inputs']['action_mask'].sum()==40*14
model.train()
with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**copy.deepcopy(batch))['loss']
assert torch.isfinite(loss)
loss.backward()
report={'loss':float(loss.detach()),'mask_valid_elements':int(batch['inputs']['action_mask'].sum()),'pixel_shape':list(batch['inputs']['pixel_values'].shape),'image_grid':batch['inputs']['image_grid_thw'].tolist(),'groups':{}}
for label,prefix in [('vision','backbone.model.model.visual.'),('language','backbone.model.model.language_model.'),('ae','action_head.')]:
 ps=[x for n,x in model.named_parameters() if n.startswith(prefix) and x.requires_grad]
 report['groups'][label]={'trainable':sum(x.numel() for x in ps),'nonzero_grad_parameters':sum(x.grad is not None and bool(x.grad.count_nonzero()) for x in ps)}
 assert report['groups'][label]['nonzero_grad_parameters']>0
model.zero_grad(set_to_none=True);model.eval()
batch['inputs'].pop('action', None)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model.get_action(**copy.deepcopy(batch))['action_pred']
assert torch.isfinite(pred).all()
report['prediction_shape']=list(pred.shape);report['peak_gib']=torch.cuda.max_memory_allocated()/2**30
(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
