"""Pure geometric relations between bounding boxes (no extra neural network).

These are *spatial observations*, not proof of a physical interaction, handover,
count, data entry, or safe storage. The same relation needs consecutive frames.
"""
from collections import Counter
from math import hypot

PAIRS = [
    ('cash','counting_machine','cash_near_counting_machine'),
    ('cash','safe','cash_near_safe'),
    ('cash','person','cash_near_person'),
    ('document','person','document_near_person'),
    ('cash_bag','person','cashbag_near_person'),
    ('person','laptop_open','person_near_laptop'),
    ('phone','person','phone_near_person')
]


def bbox_iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1]); area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    union=area_a+area_b-inter
    return inter/union if union>0 else 0.


def relative_distance(a,b):
    ca=((a[0]+a[2])/2,(a[1]+a[3])/2)
    cb=((b[0]+b[2])/2,(b[1]+b[3])/2)
    da=max(1,hypot(a[2]-a[0],a[3]-a[1]));db=max(1,hypot(b[2]-b[0],b[3]-b[1]))
    return hypot(ca[0]-cb[0],ca[1]-cb[1])/max(da,db)


def relations_on_frame(objects,max_relative_distance=.8):
    found=[]
    for a,b,name in PAIRS:
        candidates_a=[o for o in objects if o['class_name']==a and o.get('conf',0)>=.4]
        candidates_b=[o for o in objects if o['class_name']==b and o.get('conf',0)>=.4]
        if a==b:continue
        for x in candidates_a:
            for y in candidates_b:
                iou=bbox_iou(x['bbox'],y['bbox'])
                distance=relative_distance(x['bbox'],y['bbox'])
                if iou>=.015 or distance<=max_relative_distance:
                    found.append({'relation':name,'relative_distance':round(float(distance),3),
                                  'iou':round(float(iou),3),'confidence':round(min(x['conf'],y['conf']),3)})
    return found


def relation_events(samples,min_frames=2):
    """Consecutive co-occurrences become a candidate spatial interval."""
    active={};out=[]
    for index,sample in enumerate(samples+[{'t':samples[-1]['t']+1 if samples else 0,'objects':[]}]):
        frame={x['relation']:x for x in relations_on_frame(sample['objects'])}
        for name in list(active):
            if name not in frame:
                rec=active.pop(name)
                if rec['frames']>=min_frames:
                    out.append({'action':name,'start_sec':round(rec['start'],2),
                        'end_sec':round(rec['last'],2),'confidence':round(rec['conf']/rec['frames'],3),
                        'source':'box_geometry',
                        'warning':'Пространственная близость; это не доказательство выполнения действия'})
        for name,x in frame.items():
            if name not in active:
                active[name]={'start':sample['t'],'last':sample['t'],'conf':0.,'frames':0}
            active[name]['last']=sample['t'];active[name]['frames']+=1
            active[name]['conf']+=x['confidence']
    return sorted(out,key=lambda x:x['start_sec'])
