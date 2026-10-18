"""Conservative procedure classifier and candidate rule engine.

CV detections are observations, not legal proof. Rule catalog distinguishes normative
constraints from the team's proposed internal workflow hypotheses.
"""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

PROCESS_NAMES = {
    0: 'Приём наличных от клиента',
    1: 'Внутренний пересчёт наличных',
    2: 'Вход в кассовое помещение',
    3: 'Выход из кассового помещения',
    4: 'Приём и обработка упаковки с наличными',
}

ALIASES = {
    'cash': 'cash', 'cash_bag': 'cash_bag', 'cashbag': 'cash_bag',
    'counting_machine': 'counting_machine', 'person': 'person',
    'id': 'document', 'document': 'document', 'phone': 'phone',
    'open_laptop': 'laptop_open', 'close_laptop': 'laptop_closed',
    'open_door': 'door_open', 'close_door': 'door_closed',
    'safe': 'safe', 'other': 'other'
}


def clean_name(name):
    return ALIASES.get(str(name).strip().lower().replace(' ', '_'), str(name).strip().lower().replace(' ', '_'))


# Canonical action names + their Russian labels (shared with the website).
ACTION_CANON = {
    'close door': 'door_closed', 'open door': 'door_open', 'close_door': 'door_closed', 'open_door': 'door_open',
    'close_laptop': 'laptop_closed', 'open_laptop': 'laptop_open',
    'safe_closing': 'safe_closing', 'cash_counting_machine': 'cash_counting_machine', 'cash_storage': 'cash_storage',
    'cash_handover': 'cash_handover', 'cashbag_handover': 'cashbag_handover', 'computer_interaction': 'computer_interaction',
    'document_handover': 'document_handover', 'cashier_leaving': 'cashier_leaving',
}
ACTION_RU = {
    'document_handover': 'проверка/передача документов', 'cash_handover': 'передача наличных',
    'cash_counting_machine': 'пересчёт на машинке', 'computer_interaction': 'запись в компьютер',
    'cash_storage': 'уборка наличных в хранилище', 'safe_closing': 'закрытие сейфа',
    'cashier_leaving': 'уход кассира', 'cashbag_handover': 'передача инкассаторского пакета',
    'door_open': 'открытие двери', 'door_closed': 'закрытие двери',
    'laptop_open': 'открытие ноутбука', 'laptop_closed': 'закрытие ноутбука',
}


def canon_action(a):
    k = str(a).strip().lower().replace(' ', '_')
    return ACTION_CANON.get(k, k)


def _smooth_counts(counts, radius=2):
    """Median filter rejects single-frame misses; no scipy requirement."""
    if not counts: return []
    return [int(sorted(counts[max(0, i-radius):min(len(counts), i+radius+1)])[len(counts[max(0, i-radius):min(len(counts), i+radius+1)])//2]) for i in range(len(counts))]


def classify_process(samples, action_events=None):
    """Each sample: {t,objects:[{class_name,conf,bbox}]}; returns scores + explanation.

    Bag class 4 is now from *object detector* (not HSV); entry/exit only if reliable
    occupancy transition at ends. Recognised actions (from the action model) nudge the
    scores toward the process they belong to. Never use folder name or ground-truth label.
    """
    if not samples:
        return {'class_id': None, 'name': 'Недостаточно данных', 'confidence': 0.0, 'scores': {}, 'reason': 'Нет кадров'}
    counts = [sum(o['class_name']=='person' and o.get('conf',1)>=.35 for o in s['objects']) for s in samples]
    counts = _smooth_counts(counts)
    n=len(samples)
    first=counts[:max(2, n//5)]
    last=counts[-max(2,n//5):]
    one_ratio=sum(c==1 for c in counts)/n
    two_ratio=sum(c>=2 for c in counts)/n
    any_ratio=sum(c>=1 for c in counts)/n
    start_present=sum(c>=1 for c in first)/len(first)
    end_present=sum(c>=1 for c in last)/len(last)
    bag_ratio=sum(any(o['class_name']=='cash_bag' and o.get('conf',1)>=.45 for o in s['objects']) for s in samples)/n
    machine_ratio=sum(any(o['class_name']=='counting_machine' for o in s['objects']) for s in samples)/n
    # Scores are interpretable heuristic ranking, NOT calibrated probabilities.
    scores = {
        0: 0.05+0.86*two_ratio,
        1: 0.05+0.55*one_ratio+0.3*machine_ratio,
        2: 0.04+0.91*max(0, end_present-start_present) if start_present<=.35 and end_present>=.65 else 0.04,
        3: 0.04+0.91*max(0, start_present-end_present) if start_present>=.65 and end_present<=.35 else 0.04,
        4: 0.04+1.3*bag_ratio,
    }
    # Very brief bag glimpse can be false positive; minimum persistence at 2 FPS.
    # A short but repeated observation of cash_bag is more distinctive than
    # the generic 'one person' condition. Do not use bag color.
    bag_frames=sum(any(o['class_name']=='cash_bag' and o.get('conf',1)>=.45 for o in s['objects']) for s in samples)
    if bag_frames >= 2 and bag_ratio >= .015: scores[4]=1.25+bag_ratio
    else: scores[4]=0.04
    # Recognised actions nudge scores toward the process they belong to (additive, interpretable).
    acts={canon_action(e.get('action')) for e in (action_events or [])}
    if 'cashbag_handover' in acts: scores[4]+=0.6
    if two_ratio>=.15 and ('cash_handover' in acts or 'document_handover' in acts): scores[0]+=0.35
    if ('cash_counting_machine' in acts or 'cash_storage' in acts) and two_ratio<.15: scores[1]+=0.3
    if 'cashier_leaving' in acts: scores[3]+=0.4
    if any_ratio<.05 and bag_ratio<.015 and not acts:
        return {'class_id':None,'name':'Недостаточно данных','confidence':0.,
                'scores':{str(k):round(float(v),3) for k,v in scores.items()},
                'reason':'Детектор не нашёл людей или характерных объектов'}
    best=max(scores,key=scores.get)
    ordered=sorted(scores.values(),reverse=True)
    margin=(ordered[0]-ordered[1]) if len(ordered)>1 else 0
    confidence=round(min(0.95,max(0.15,0.45+margin*0.45)),3)
    parts=[f'человек в кадре: {any_ratio:.0%}',f'двое или более: {two_ratio:.0%}',
           f'пакет cash_bag: {bag_ratio:.0%}', f'счётная машина: {machine_ratio:.0%}',
           f'присутствие в начале/конце: {start_present:.0%}/{end_present:.0%}']
    if acts:
        parts.append('распознанные действия: '+', '.join(ACTION_RU.get(a,a) for a in sorted(acts)))
    return {'class_id':int(best),'name':PROCESS_NAMES[best], 'confidence':confidence,
            'scores':{str(k):round(float(v),3) for k,v in scores.items()},'reason':'; '.join(parts),
            'warning':'Классификация процесса эвристическая; confidence не является вероятностью правильности.'}


def _obj_present(s, label, min_conf=.35):
    return any(o['class_name']==label and o.get('conf',1)>=min_conf for o in s['objects'])


def detection_events(samples, min_duration=.35):
    """Event/state candidates from model's detector labels, not invented actions.
    Distinguish close_door label (appearance) from observed transition (stronger).
    """
    events=[]
    for label, action in [('door_open','door_open'),('door_closed','door_closed'),
                          ('laptop_open','laptop_open'),('laptop_closed','laptop_closed')]:
        starts=None; last_t=None; acc=[]
        for s in samples+[{'t': (samples[-1]['t']+1) if samples else 0,'objects':[]}]:
            present=_obj_present(s,label,.4)
            if present:
                if starts is None: starts=s['t']; acc=[]
                last_t=s['t'];acc.append(max(o.get('conf',1) for o in s['objects'] if o['class_name']==label))
            elif starts is not None:
                if last_t-starts>=min_duration:
                    events.append({'action':action,'start_sec':round(starts,2),'end_sec':round(last_t,2),
                                   'confidence':round(float(sum(acc)/len(acc)),3),'source':'detector_state',
                                   'evidence':'Обнаружен объект с соответствующей меткой состояния; не доказана системная блокировка/запирание'})
                starts=None
    return sorted(events,key=lambda e:e['start_sec'])


def candidate_checks(process, samples, events, rules_catalog):
    """Only report highly observable warnings; do NOT infer absent actions from crop/truncation.

    All emitted checks are candidate procedural hypotheses unless direct norm can
    be linked with adequate evidence. No automatic statement of legal guilt.
    """
    warnings=[]
    if not samples: return warnings
    duration=samples[-1]['t']
    trailing=samples[-max(3,round(len(samples)*.15)):]
    def ratio(c): return sum(_obj_present(s,c,.45) for s in trailing)/len(trailing)
    person_last = sum(any(o['class_name']=='person' for o in s['objects']) for s in trailing)/len(trailing)
    # Report only if the departure is supported by the *process* classifier.
    if process.get('class_id') == 3 and person_last < .25:
        if ratio('cash')>=.45:
            warnings.append({'code':'H02_CASH_LEFT_ON_DESK', 'type':'Возможное оставление наличных при уходе',
                            'timecode_sec':round(duration,2),
                            'description':'После исчезновения кассира камера продолжает видеть объект, классифицированный как наличные. Местоположение (на столе или в хранилище) требует проверки человеком.',
                            'source_status':'hypothesis', 'rule_id':'H02', 'confidence':.55})
        if ratio('laptop_open')>=.45 and ratio('laptop_closed')<.2:
            warnings.append({'code':'H02_LAPTOP_LEFT_OPEN', 'type':'Возможное незавершённое закрытие ноутбука',
                            'timecode_sec':round(duration,2),
                            'description':'При уходе кассира ноутбук визуально выглядит открытым. Это не доказывает, что рабочая сессия не заблокирована.',
                            'source_status':'hypothesis','rule_id':'H02','confidence':.6})
        if ratio('door_open')>=.45 and ratio('door_closed')<.2:
            warnings.append({'code':'H02_DOOR_LEFT_OPEN','type':'Возможно оставлена открытой дверь помещения',
                            'timecode_sec':round(duration,2),
                            'description':'После ухода наблюдаются кадры с открытой дверью. Наличие обязательного требования её закрыть должно быть подтверждено регламентом.',
                            'source_status':'hypothesis','rule_id':'H02','confidence':.6})
    # Machine+cash co-visible is evidence of candidate action, NOT confirmation of count.
    return warnings


def rules_from_file(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    return data


def check_order_hypotheses(process, events, graphs, min_conf=.78):
    """Find only *observed, separated*, reversed steps; never flag mere absence.

    Actions trained with whole-video labels produce approximate windows; these are
    low-confidence research cues only, not verified banking-rule violations.
    """
    key=str(process.get('class_id'))
    if key not in graphs: return []
    graph=graphs[key]
    mapping={'close door':'door_closed','open door':'door_open',
             'safe_closing':'safe_closing', 'cash_counting_machine':'cash_counting_machine',
             'cash_storage':'cash_storage','cash_handover':'cash_handover',
             'cashbag_handover':'cashbag_handover',
             'computer_interaction':'computer_interaction',
             'document_handover':'document_handover','cashier_leaving':'cashier_leaving',
             'close_door':'door_closed','open_door':'door_open',
             'close_laptop':'laptop_closed','open_laptop':'laptop_open'}
    reliable=[]
    for e in events:
        name=mapping.get(e['action'].strip().lower().replace(' ','_'),e['action'].strip().lower().replace(' ','_'))
        if e['confidence']>=min_conf:
            reliable.append({**e,'normalized':name})
    flagged=[]
    for before,after in graph.get('order_edges',[]):
        early=[e for e in reliable if e['normalized']==before]
        late=[e for e in reliable if e['normalized']==after]
        if not early or not late:continue
        # If a single reverse pair exists with disjoint intervals and no correctly
        # ordered pair exists, flag it for manual review. No negative inference.
        has_correct=any(a['end_sec']+.2<=b['start_sec'] for a in early for b in late)
        if has_correct:continue
        reverse=next(((a,b) for a in early for b in late
                      if b['end_sec']+.5<a['start_sec']),None)
        if reverse is None: continue
        a,b=reverse
        flagged.append({'code':f'SEQUENCE_{before}_{after}',
                        'type':'Возможное нарушение порядка действий',
                        'timecode_sec':round(b['start_sec'],2),
                        'description':f'Распознано действие «{after}» раньше «{before}». '
                            'Это несоответствие предполагаемому порядку процесса. '
                            'Границы действий предварительные и требуют проверки.',
                        'source_status':'hypothesis',
                        'rule_id':'procedure_graph_'+key,
                        'confidence':round(min(a['confidence'],b['confidence'])*.65,3)})
    return flagged


def missing_step_checks(process, action_events, graphs):
    """Flag a required step as skipped ONLY when it is missing *inside* the observed
    sequence: there is a recognised required action both before and after it in the
    expected order. A missing step at the very start/end is NOT flagged, because the
    clip may simply be truncated there (we did not see that part of the process).
    """
    cid = process.get('class_id')
    if cid is None or str(cid) not in graphs:
        return []
    required = graphs[str(cid)].get('required_actions') or []
    if len(required) < 3:
        return []  # need room for a step to sit *between* observed neighbours
    # Earliest timecode per recognised canonical action.
    observed = {}
    for e in action_events or []:
        c = canon_action(e.get('action'))
        if c not in observed or e['start_sec'] < observed[c]['start_sec']:
            observed[c] = e
    obs_idx = [i for i, a in enumerate(required) if a in observed]
    if not obs_idx:
        return []  # action model saw none of the required steps — no anchor, stay silent
    findings = []
    for i, a in enumerate(required):
        if a in observed:
            continue
        before = [j for j in obs_idx if j < i]
        after = [j for j in obs_idx if j > i]
        if not before or not after:
            continue  # edge step — possible clip truncation, do not flag
        ru = ACTION_RU.get(a, a)
        t_ref = observed[required[after[0]]]['start_sec']
        findings.append({'code': f'MISSING_{a}',
                         'type': f'Пропущен обязательный шаг: {ru}',
                         'timecode_sec': round(float(t_ref), 2),
                         'description': f'В процессе «{process.get("name")}» шаг «{ru}» не распознан, '
                             f'хотя наблюдались действия и до, и после него — то есть фрагмент в этом месте '
                             f'не обрезан. Похоже на пропуск шага. Требует проверки человеком.',
                         'source_status': 'hypothesis',
                         'rule_id': 'required_' + str(cid),
                         'confidence': 0.6})
    return findings


def _max_people(samples):
    counts=[sum(o['class_name']=='person' and o.get('conf',1)>=.4 for o in s['objects']) for s in samples]
    counts=_smooth_counts(counts)
    return max(counts) if counts else 0


def _ever_detected(samples,label,min_conf=.4,min_frames=1):
    return sum(_obj_present(s,label,min_conf) for s in samples)>=min_frames


def object_condition_checks(process, samples, action_events, graphs):
    """Object/occupancy-based checks that do NOT depend on the (weak) action model:
    missing characteristic objects, extra people, phone. Each is a candidate hypothesis.
    """
    cid=process.get('class_id')
    if cid is None or not samples:
        return []
    out=[]
    duration=samples[-1]['t']
    maxp=_max_people(samples)
    # For cash-handling processes, no confirmed action step at all is itself suspicious.
    if cid in (0,1,4):
        required=(graphs.get(str(cid)) or {}).get('required_actions') or []
        obs={canon_action(e.get('action')) for e in (action_events or [])}
        if required and not any(a in obs for a in required):
            out.append({'code':f'ACTIONS_UNCONFIRMED_{cid}','type':'Ключевые действия процесса не подтверждены',
                'timecode_sec':round(duration,2),
                'description':f'Процесс классифицирован как «{process.get("name")}», но модель действий не подтвердила '
                    'ни одного характерного шага этого процесса. Классификация ненадёжна или процедура неполная/вне кадра.',
                'source_status':'hypothesis','rule_id':'actions_unconfirmed_'+str(cid),'confidence':.5})
    if cid==0:  # приём наличных от клиента
        if not _ever_detected(samples,'document',.4,1):
            out.append({'code':'NO_DOCUMENT','type':'Документы клиента не наблюдались',
                'timecode_sec':round(duration,2),
                'description':'Процесс похож на приём наличных, но документ ни разу не обнаружен в кадре — '
                    'возможно, документы клиента не проверялись (либо были вне поля камеры).',
                'source_status':'hypothesis','rule_id':'C01','confidence':.55})
        if maxp>2:
            out.append({'code':'TOO_MANY_PEOPLE','type':'Более двух человек в кассовой зоне',
                'timecode_sec':round(duration,2),
                'description':f'Одновременно в кадре до {maxp} человек; для приёма наличных ожидаются только кассир и клиент.',
                'source_status':'hypothesis','rule_id':'C03','confidence':.6})
    if cid==1:  # внутренний пересчёт
        if maxp>=2:
            out.append({'code':'EXTRA_PERSON','type':'Посторонний во время внутреннего пересчёта',
                'timecode_sec':round(duration,2),
                'description':f'Во время внутреннего пересчёта в кадре одновременно до {maxp} человек, тогда как ожидается один сотрудник.',
                'source_status':'hypothesis','rule_id':'C09','confidence':.55})
        if _ever_detected(samples,'phone',.4,2):
            out.append({'code':'PHONE_PRESENT','type':'Обнаружен телефон во время пересчёта',
                'timecode_sec':round(duration,2),
                'description':'В кадре во время пересчёта наблюдается телефон.',
                'source_status':'hypothesis','rule_id':'H01','confidence':.5})
    return out
