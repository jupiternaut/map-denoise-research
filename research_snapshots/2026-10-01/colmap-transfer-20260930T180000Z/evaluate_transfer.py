"""Post-seal evaluation; fixed query support, no GT routing or alignment."""
import csv
import numpy as np
from scipy.spatial import cKDTree
from mvs_transfer import ROOT,load,dump,npz,sha,verify,verify_lock

def read_ply_xyz(path):
    scalar={'char':'i1','int8':'i1','uchar':'u1','uint8':'u1','short':'i2','int16':'i2',
            'ushort':'u2','uint16':'u2','int':'i4','int32':'i4','uint':'u4','uint32':'u4',
            'float':'f4','float32':'f4','double':'f8','float64':'f8'}
    with open(path,'rb') as f:
        assert f.readline().strip()==b'ply'; fields=[]; n=None; fmt=None; element=None
        for _ in range(1000):
            line=f.readline(); assert line
            p=line.decode('ascii').strip().split()
            if not p:continue
            if p[0]=='format':fmt=p[1]
            elif p[0]=='element':
                element=p[1]
                if element=='vertex':n=int(p[2])
            elif p[0]=='property' and element=='vertex':fields.append((p[2],scalar[p[1]]))
            elif p[0]=='end_header':break
        else:raise ValueError('PLY header too long')
        assert n and all(v in dict(fields) for v in ('x','y','z'))
        if fmt=='ascii':
            names=[x[0] for x in fields]; xyz=np.loadtxt(f,max_rows=n,usecols=[names.index(x) for x in ('x','y','z')])
        else:
            assert fmt in ('binary_little_endian','binary_big_endian')
            endian='<' if fmt=='binary_little_endian' else '>'
            a=np.fromfile(f,dtype=np.dtype([(name,endian+t) for name,t in fields]),count=n)
            xyz=np.column_stack([a[x] for x in ('x','y','z')]).astype(float)
    assert xyz.shape==(n,3) and np.isfinite(xyz).all(); return xyz
def metrics(d,mask):
    x=d[mask]
    return dict(n=int(len(x)),mse_mm2=float(np.mean(x*x)) if len(x) else None,
        mae_mm=float(np.mean(x)) if len(x) else None,median_mm=float(np.median(x)) if len(x) else None,
        correct_1mm=int((x<=1).sum()),wrong_5mm=int((x>5).sum()))
def summarize(rows,method):
    rs=[r for r in rows if r['method']==method and r['scope']=='cpu_valid_common']
    mean=lambda key:float(np.mean([r[key] for r in rs])) if all(r[key] is not None for r in rs) else None
    return dict(method=method,roi_equal_mse_mm2=mean('mse_mm2'),roi_equal_mae_mm=mean('mae_mm'),
       common_n=sum(r['n'] for r in rs),correct_1mm=sum(r['correct_1mm'] for r in rs),wrong_5mm=sum(r['wrong_5mm'] for r in rs))
def main():
    verify_lock(); seal=load(ROOT/'PREDICTIONS_SEALED.json'); verify(seal['files'])
    manifest=load(ROOT/'references/MANIFEST.json'); assert manifest['prediction_seal_sha256']==sha(ROOT/'PREDICTIONS_SEALED.json')
    refs={r['scene']:r for r in manifest['files']}; plan=load(ROOT/'PLAN.json')
    rows=[]; points=[]; tail=[]; brute=[]
    for sid,s in plan['scenes'].items():
        ref=refs[int(sid)]; assert sha(ref['path'])==ref['sha256']; gt=read_ply_xyz(ref['path']); tree=cKDTree(gt)
        for roi in s['rois']:
            name=roi['id']; cpu=npz(ROOT/'cpu'/(name+'.npz')); new=npz(ROOT/'predictions'/(name+'.npz'))
            uv=np.array(roi['pixel_xy']); assert np.array_equal(cpu['pixel_xy'],uv) and np.array_equal(new['pixel_xy'],uv)
            valid=cpu['valid']; assert np.array_equal(valid,np.isfinite(cpu['xyz_mm']).all(1))
            kd=np.full(len(uv),np.nan); kd[valid]=tree.query(cpu['xyz_mm'][valid],workers=1)[0]
            distances={'CPU':kd}; masks={'CPU':valid}
            for arm in ['photo','geo']:
                v=new[arm+'_valid']; xyz=new[arm+'_xyz_mm']; assert np.array_equal(v,np.isfinite(xyz).all(1))
                d=np.full(len(uv),np.nan); d[v]=tree.query(xyz[v],workers=1)[0]
                distances[arm]=d; masks[arm]=v
                fb=kd.copy(); fb[valid&v]=d[valid&v]; distances[arm+'_fallback']=fb; masks[arm+'_fallback']=valid
                oldgood=valid&(kd<=1); hard=valid&(kd>5)
                tail.append(dict(scene=int(sid),roi=name,method=arm,baseline_valid=int(valid.sum()),
                    improved=int((valid&v&(d<kd-1e-9)).sum()),worsened=int((valid&v&(d>kd+1e-9)).sum()),
                    unchanged=int((valid&(~v|np.isclose(d,kd,rtol=0,atol=1e-9))).sum()),
                    old_good_1mm=int(oldgood.sum()),old_good_harmed_beyond_1mm=int((oldgood&v&(d>1)).sum()),
                    old_wrong_5mm=int(hard.sum()),rescued_5mm=int((hard&v&(d<=5)).sum()),rescued_1mm=int((hard&v&(d<=1)).sum()),
                    newly_supported=int((~valid&v).sum()),newly_supported_correct_1mm=int((~valid&v&(d<=1)).sum())))
                if v.any():
                    j=int(np.flatnonzero(v)[0]); exact=float(np.sqrt(np.min(np.sum((gt-xyz[j])**2,axis=1))))
                    delta=abs(exact-d[j]); assert delta<1e-9
                    brute.append(dict(roi=name,method=arm,query=j,bruteforce_mm=exact,kdtree_mm=float(d[j]),difference_mm=delta))
            for method,d in distances.items():
                v=masks[method]
                for scope,mask in [('all_valid',v),('cpu_valid_common',v&valid)]:
                    stat=metrics(d,mask)
                    rows.append(dict(scene=int(sid),roi=name,method=method,scope=scope,requested=len(uv),**stat,
                        coverage=stat['n']/len(uv),correct_per_requested=stat['correct_1mm']/len(uv),
                        paired_cpu_mse_mm2=metrics(kd,mask&valid)['mse_mm2']))
                for i,(u,w) in enumerate(uv):
                    points.append(dict(scene=int(sid),roi=name,query=i,u=int(u),v=int(w),method=method,valid=bool(v[i]),
                        distance_mm=float(d[i]) if v[i] else '',cpu_valid=bool(valid[i]),cpu_distance_mm=float(kd[i]) if valid[i] else ''))
        print('EVALUATED',sid,len(gt),flush=True)
        del tree,gt
    methods=['CPU','photo_fallback','geo_fallback']; summary=[summarize(rows,m) for m in methods]
    scenes={str(sid):[summarize([r for r in rows if r['scene']==sid],m) for m in methods] for sid in (118,122)}
    output=ROOT/'evaluation'; output.mkdir(exist_ok=False)
    for name,data in [('ROI_METRICS.csv',rows),('POINT_METRICS.csv',points),('TAIL_METRICS.csv',tail)]:
        with (output/name).open('x') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    dump(output/'RESULTS.json',dict(summary=summary,scenes=scenes,tail=tail,references=refs,
        evaluation_type='real_gt; two previously unrun same-source DTU objects; not cross-sensor or official full-scene benchmark',
        fixed_query_count=512,baseline_valid=summary[0]['common_n'],gt_used_for_prediction=False,deployment_changed=False))
    dump(output/'BRUTE_FORCE_CHECK.json',brute)
    print(summary,flush=True)
if __name__=='__main__':main()
