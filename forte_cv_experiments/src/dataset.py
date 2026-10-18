import csv, hashlib, re, shutil, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import cv2

TEST_IDS = ['X-15', 'X-16', 'X-22', 'X-23']  # two positive/two negative, reserve together


def vid_id(value):
    """Both Cyrillic Х and Latin X filenames map to X-<number>."""
    m = re.search(r'\d+', str(value))
    if not m:
        raise ValueError(f'Cannot parse video ID from {value!r}')
    return f'X-{int(m.group())}'


def read_xlsx_first_sheet(file_path):
    """Minimal stdlib XLSX reader, independent of openpyxl/excel versions."""
    NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(file_path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            tree=ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in tree.findall('m:si', NS):
                shared.append(''.join(t.text or '' for t in si.findall('.//m:t', NS)))
        sheets=[x for x in z.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml',x)]
        if not sheets:raise ValueError('No worksheet in '+str(file_path))
        root=ET.fromstring(z.read(sorted(sheets)[0]))
        rows=[]
        for r in root.findall('.//m:sheetData/m:row',NS):
            result={}
            for c in r.findall('m:c',NS):
                ref=c.attrib.get('r','A1')
                letters=''.join(k for k in ref if k.isalpha())
                idx=0
                for char in letters:idx=idx*26+ord(char)-64
                el=c.find('m:v',NS)
                if c.attrib.get('t')=='inlineStr':
                    txt=''.join(t.text or '' for t in c.findall('.//m:t',NS))
                elif el is None:txt=''
                elif c.attrib.get('t')=='s':txt=shared[int(el.text)]
                else:txt=el.text or ''
                result[idx-1]=txt
            if result: rows.append([result.get(i,'') for i in range(max(result)+1)])
        return rows


def parse_sec(text):
    if not text:return ''
    s=str(text).lower().replace('ё','е')
    nums=[int(x) for x in re.findall(r'\d+',s)]
    if not nums:return ''
    if 'мин' in s and len(nums)>=2:return str(60*nums[0]+nums[1])
    if 'мин' in s:return str(60*nums[0])
    return str(nums[0])


def video_length(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError('Cannot open video '+str(path))
    frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps=cap.get(cv2.CAP_PROP_FPS)
    w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return round(frames/fps,3) if fps>0 else 0.0,int(w),int(h)


def prepare(zip_path, labels_path, out_dir, test_ids=TEST_IDS):
    out=Path(out_dir); vids=out/'videos';vids.mkdir(parents=True,exist_ok=True)
    labels={}
    for row in read_xlsx_first_sheet(labels_path)[1:]:
        if len(row)<3:continue
        name, timecode, verdict=row[:3]
        if not name:continue
        label=0 if 'без' in verdict.lower() else 1
        labels[vid_id(name)]={'label':label,'timecode_sec':parse_sec(timecode)}
    files={}
    with zipfile.ZipFile(zip_path) as z:
        for item in z.infolist():
            if not item.filename.lower().endswith(('.mov','.mp4')) or item.filename.startswith('__MACOSX/'):
                continue
            # DO NOT trust archive paths (avoid path traversal and Cyrillic filename issues)
            video=vid_id(item.filename)
            if video in files:raise ValueError('Duplicate video ID '+video)
            target=vids/f'{video}.mov'
            if not target.exists() or target.stat().st_size!=item.file_size:
                with z.open(item) as src,open(target,'wb') as dst:
                    shutil.copyfileobj(src,dst)
            files[video]=target
    if labels.keys()!=files.keys():
        raise ValueError(f'Missing label IDs: {files.keys()-labels.keys()}, missing videos: {labels.keys()-files.keys()}')
    test={vid_id(v) for v in test_ids}
    if not test.issubset(files.keys()):raise ValueError(f'Unknown test IDs: {test-files.keys()}')
    if sum(labels[i]['label'] for i in test)!=2 or len(test)!=4:
        raise ValueError('This experimental protocol requires exactly two positive and two negative test videos.')
    hash_to_id={}
    rows=[]
    for vid in sorted(files,key=lambda v:int(v.split('-')[1])):
        path=files[vid]
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        # exact duplicates get the same group and are removed from training via duplicate_of
        dup=hash_to_id.get(digest,'')
        if not dup:hash_to_id[digest]=vid
        duration,w,h=video_length(path)
        if vid in test and dup and dup not in test:raise ValueError(f'Test {vid} exactly duplicates train {dup}; choose another holdout')
        if vid not in test and dup in test:raise ValueError(f'Train {vid} exactly duplicates test {dup}; choose another holdout')
        rows.append(dict(video_id=vid,path=str(path.resolve()),split='test' if vid in test else 'train',
                         label=labels[vid]['label'],timecode_sec=labels[vid]['timecode_sec'],
                         duration_sec=duration,width=w,height=h,sha256=digest,duplicate_of=dup))
    out.mkdir(parents=True,exist_ok=True)
    dest=out/'manifest.csv'
    with open(dest,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Prepared {len(rows)} videos: {sum(x["split"]=="test" for x in rows)} held out; '
          f'{sum(bool(x["duplicate_of"]) for x in rows)} exact duplicate(s)')
    for r in rows:
        if r['duplicate_of']:
            print(f'DUPLICATE: {r["video_id"]} == {r["duplicate_of"]}; timecodes: '
                  f'{r["timecode_sec"]}, '+next(x['timecode_sec'] for x in rows if x['video_id']==r['duplicate_of']))
    print('Test IDs:', ', '.join(sorted(test)))
    return rows


def load_manifest(path):
    with open(path,encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))
