# inside app/workers.py where detections are processed

from app.storage import save_full_frame
import os
import json
from datetime import datetime
# ... other imports

# inside detection loop, replace crop logic with:
for d in detections:
    bbox = d['bbox']  # [x1,y1,x2,y2]
    centroid = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)
    roi_hit = True
    if roi:
        roi_hit = bbox_intersects_roi(tuple(bbox), roi)
    if not roi_hit:
        continue

    detection_id = str(uuid.uuid4())
    # save entire frame, returns relative path under STORAGE_BASE
    rel_image_path = save_full_frame(frame, detection_id, camera_id=camera_id, camera_name=camera_row.get('camera_name'))

    # Build DB record, store relative path in image_path
    rec = {
        'detection_id': detection_id,
        'camera_id': camera_id,
        'detected_class': d.get('class'),
        'confidence': float(d.get('confidence', 0)),
        'bbox': json.dumps({'x1':bbox[0],'y1':bbox[1],'x2':bbox[2],'y2':bbox[3]}),
        'centroid': json.dumps({'x':centroid[0],'y':centroid[1]}),
        'roi_hit': 1,
        'image_path': rel_image_path,
        'passage_time': datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'),
        'inference_ms': latency,
        'extra': json.dumps({'roi': roi if roi is not None else None})
    }

    try:
        with SessionLocal() as session:
            sql = """INSERT INTO atcc_records
                     (detection_id,camera_id,detected_class,confidence,bbox,centroid,roi_hit,image_path,passage_time,inference_ms,extra)
                     VALUES (:detection_id,:camera_id,:detected_class,:confidence,:bbox,:centroid,:roi_hit,:image_path,:passage_time,:inference_ms,:extra)"""
            session.execute(sql, rec)
            session.commit()
    except Exception as e:
        print('DB write failed', e)
