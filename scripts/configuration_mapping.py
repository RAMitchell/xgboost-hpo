from collector import np,logspace,spike
def design(seed, count, prefix):
    unit=np.random.default_rng(seed).random((count,13))
    out=[]
    for i,u in enumerate(unit):
        mode=['depth','leaves','both'][min(int(u[4]*3),2)]
        depth=int(2+u[1]*11);leaves=int(round(logspace(u[2],8,256)))
        # A finite leaf cap safeguards every depth-unlimited configuration.
        p=dict(learning_rate=logspace(u[0],.01,.3),max_depth=depth if mode!='leaves' else 0,
               max_leaves=leaves if mode!='depth' else 0,grow_policy=['depthwise','lossguide'][int(u[3]*2)],
               subsample=1. if u[5]<.125 else .5+.5*(u[5]-.125)/.875,
               min_child_weight=logspace(u[8],1e-5,100),reg_lambda=spike(u[9],100),reg_alpha=spike(u[10],10),gamma=spike(u[11],10),
               max_bin=[128,256,512][min(int(u[12]*3),2)],colsample_bytree=1.,colsample_bylevel=1.,colsample_bynode=1.)
        cm=['tree','level','node'][min(int(u[6]*3),2)]
        p['colsample_by'+cm]=1. if u[7]<.125 else .5+.5*(u[7]-.125)/.875
        out.append(dict(cid=f'{prefix}_{i:03}',group='shared_iid',base_index=i,size_mode=mode,column_mode=cm,params=p,unit=u.tolist()))
    return out
