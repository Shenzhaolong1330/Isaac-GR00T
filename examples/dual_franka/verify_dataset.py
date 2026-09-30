"""Verify converted video alignment and produce a held-out hardware-free fixture."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from torchcodec.decoders import VideoDecoder

def main():
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--converted',type=Path,required=True);a.add_argument('--episodes',type=Path,required=True);a.add_argument('--output',type=Path,required=True);p=a.parse_args()
    info=json.loads((p.source/'meta/info.json').read_text());original={x['episode_index']:x for x in json.loads(p.episodes.read_text())}
    # Validation loading uses the training-only statistics, never its own fit.
    shutil.copy2(p.converted/'train/meta/stats.json',p.converted/'validation/meta/stats.json')
    records=[];fixture=None
    for part in ['train','validation']:
        eps=[json.loads(s) for s in (p.converted/part/'meta/episodes.jsonl').read_text().splitlines()]
        chosen={0,len(eps)-1};groups={}
        for i,e in enumerate(eps):groups.setdefault((e['source'],tuple(e['tasks'])),i)
        chosen.update(groups.values())
        for i in sorted(chosen):
            ep=eps[i];old=original[ep['source_episode_index']];n=ep['length'];indices=[0,n//2,n-1]
            for camera in ['head_image','left_wrist_image','right_wrist_image']:
                key='observation.images.'+camera
                src=p.source/info['video_path'].format(video_key=key,chunk_index=old[f'videos/{key}/chunk_index'],file_index=old[f'videos/{key}/file_index'])
                dst=p.converted/part/f'videos/chunk-{i//1000:03d}/{key}/episode_{i:06d}.mp4'
                source_decoder=VideoDecoder(str(src),dimension_order='NHWC',num_ffmpeg_threads=1)
                output_decoder=VideoDecoder(str(dst),dimension_order='NHWC',num_ffmpeg_threads=1)
                offset=round(old[f'videos/{key}/from_timestamp']*30)
                x=source_decoder.get_frames_at(indices=[offset+j for j in indices]).data.numpy()
                y=output_decoder.get_frames_at(indices=indices).data.numpy()
                error=np.abs(x.astype(np.int16)-y.astype(np.int16))
                assert len(output_decoder)==n
                assert error.max()<=2,(part,i,camera,error.max(),error.mean())
                records.append({'partition':part,'episode':i,'camera':camera,'indices':indices,'max_rgb_difference':int(error.max()),'mae':float(error.mean())})
                if part=='validation' and i==0:
                    if fixture is None:fixture={}
                    fixture[{'head_image':'head','left_wrist_image':'left_wrist','right_wrist_image':'right_wrist'}[camera]]=y[0]
            if part=='validation' and i==0:
                table=pq.read_table(p.converted/part/f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet')
                fixture['state']=np.array(table['observation.state'][0].as_py(),dtype=np.float32)
                tid=int(table['task_index'][0].as_py());tasks={t['task_index']:t['task'] for t in map(json.loads,(p.converted/part/'meta/tasks.jsonl').read_text().splitlines())}
                fixture['text']=np.array(tasks[tid]);fixture['target']=np.array(table['action'][:40].to_pylist(),dtype=np.float32)
    p.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p.output/'replay_fixture.npz',**fixture)
    (p.output/'video_alignment.json').write_text(json.dumps(records,indent=2));print('verified',len(records),'video boundary samples',flush=True)
if __name__=='__main__':main()
