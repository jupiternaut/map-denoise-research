"""Paired official gsplat renders: original vs unchanged v1 opacity baseline."""
import sys,json,time,math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/srv/slam-research/grf/map-denoise/tools/published-plyfile')
from plyfile import PlyData
from PIL import Image,ImageDraw
from gsplat import rasterization
DATA=Path('/srv/slam-research/grf/map-denoise/datasets/published-outputs-v1/room')
V1=Path('/srv/slam-research/grf/map-denoise/runs/published-outputs-v1/room')
OUT=Path('/srv/slam-research/grf/map-denoise/runs/published-outputs-v2/room')

def load(path):
    a=PlyData.read(str(path))['vertex'].data
    def cols(keys):return np.column_stack([a[k] for k in keys]).astype(np.float32)
    means=cols(['x','y','z']);q=cols([f'rot_{j}' for j in range(4)])
    scales=np.exp(cols([f'scale_{j}' for j in range(3)]));op=1/(1+np.exp(-np.asarray(a['opacity'])))
    sh=np.concatenate((cols([f'f_dc_{j}' for j in range(3)])[:,None,:],
        cols([f'f_rest_{j}' for j in range(45)]).reshape(-1,3,15).transpose(0,2,1)),axis=1)
    return tuple(torch.from_numpy(np.ascontiguousarray(x)).to('cuda') for x in (means,q,scales,op,sh))

def camera(c,width=1024):
    h=round(c['height']*width/c['width']);sx=width/c['width'];sy=h/c['height']
    r=np.asarray(c['rotation']);pos=np.asarray(c['position']);view=np.eye(4);view[:3,:3]=r.T;view[:3,3]=-r.T@pos
    k=np.array([[c['fx']*sx,0,width/2],[0,c['fy']*sy,h/2],[0,0,1]])
    np.testing.assert_allclose(view[:3,:3]@pos+view[:3,3],0,atol=1e-8)
    return torch.tensor(view[None],device='cuda',dtype=torch.float32),torch.tensor(k[None],device='cuda',dtype=torch.float32),width,h

@torch.no_grad()
def render(model,c):
    v,k,w,h=camera(c);torch.cuda.synchronize();start=time.perf_counter()
    rgb,alpha,meta=rasterization(*model,v,k,w,h,sh_degree=3,packed=True,near_plane=.01,far_plane=1e10,
        radius_clip=0.,rasterize_mode='classic',backgrounds=None)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    rgb=rgb[0].clamp(0,1).cpu().numpy();alpha=alpha[0].cpu().numpy()
    assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
    return rgb,alpha,seconds

def png(path,x):Image.fromarray(np.uint8(np.clip(x,0,1)*255+.5)).save(path)

def main():
    OUT.mkdir(parents=True,exist_ok=True);cameras=json.loads((DATA/'cameras.json').read_text());ids=[0,103,206,310]
    original=load(DATA/'point_cloud/iteration_30000/point_cloud.ply')
    # Compile and warm up on a single Gaussian before timing full model renders.
    render(tuple(x[:1] for x in original),cameras[0])
    baseline={};metrics=[]
    for i in ids:
        rgb,a,t=render(original,cameras[i]);baseline[i]=(rgb,a,t);png(OUT/f'camera{i}_original.png',rgb)
    repeat,_,_=render(original,cameras[0]);repeat_error=float(np.max(np.abs(repeat-baseline[0][0])))
    del original;torch.cuda.empty_cache()
    candidate=load(V1/'room_opacity001_baseline.ply')
    for i in ids:
        rgb,a,t=render(candidate,cameras[i]);old,oa,ot=baseline[i];diff=np.abs(rgb-old);mse=float(np.mean(diff**2))
        row=dict(camera_index=i,img_name=cameras[i]['img_name'],width=rgb.shape[1],height=rgb.shape[0],
            mean_absolute_rgb_difference=float(diff.mean()),p99_absolute_rgb_difference=float(np.quantile(diff,.99)),
            changed_pixel_fraction_gt_1_over_255=float(np.mean(diff.max(2)>1/255)),
            original_reference_psnr_db=-10*math.log10(mse) if mse>0 else None,
            mean_alpha_change=float((a-oa).mean()),original_render_seconds=ot,pruned_render_seconds=t)
        metrics.append(row);png(OUT/f'camera{i}_pruned.png',rgb);png(OUT/f'camera{i}_difference_x10.png',diff*10)
        canvas=Image.new('RGB',(rgb.shape[1]*3,rgb.shape[0]+32),'white')
        for j,(name,arr) in enumerate((('Original',old),('Opacity < 0.01 pruned',rgb),('Absolute difference x10',diff*10))):
            canvas.paste(Image.fromarray(np.uint8(np.clip(arr,0,1)*255+.5)),(j*rgb.shape[1],32))
            ImageDraw.Draw(canvas).text((j*rgb.shape[1]+5,8),name,fill='black')
        canvas.save(OUT/f'camera{i}_comparison.jpg',quality=95)
        print(row,flush=True)
    result=dict(metrics=metrics,original_repeat_max_absolute_error=repeat_error,torch=torch.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),gpu_max_allocated_bytes=torch.cuda.max_memory_allocated(),
        gsplat_version='1.5.3',scope='paired renders only; no original photos, no GT image quality claim',
        settings=dict(sh_degree=3,rasterize_mode='classic',radius_clip=0,width=1024,background='black',camera_indices=ids))
    (OUT/'RENDER_RESULTS.json').write_text(json.dumps(result,indent=2));print('RENDER COMPLETE',flush=True)
if __name__=='__main__':main()
