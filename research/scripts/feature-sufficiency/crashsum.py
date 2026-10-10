import json,sys,collections
for fn in sys.argv[1:]:
  for l in open(fn):
    e=json.loads(l)
    if e['out']!='crash': continue
    r=e['ring']; last=r[-1]
    # find takeoff: last frame where jumping false followed by true
    tk=None
    for i in range(len(r)-1,0,-1):
      if r[i]['j'] and not r[i-1]['j']: tk=i;break
    s30=e.get('snaps',{}).get('30',{}).get('fs',{})
    s12=e.get('snaps',{}).get('12',{}).get('fs',{})
    nx=s12.get('next')
    print(fn.split('/')[-1][:5], e['seed'],e.get('level'),e['score'],e['type'],e.get('size'),'yc',e.get('yClass'),'v',e.get('v'),'sp',e.get('speed'),
      'dino',e['crashDino'],'takeoff_ago',None if tk is None else len(r)-1-tk,
      'req12',s12.get('req'),'acts',''.join(str(x['a']) for x in r[-30:]),
      'next',nx and (nx['t'],nx['visible'],nx['tau'],nx['gapPx']))
