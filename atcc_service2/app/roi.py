from shapely.geometry import Point, Polygon, box
from typing import List, Tuple


# ROI helpers. ROI is expected as JSON structure: {"type":"polygon","points":[[x1,y1],[x2,y2],...]} or rectangle [x1,y1,x2,y2]


def point_in_polygon(pt: Tuple[float,float], poly_points: List[Tuple[float,float]]) -> bool:
poly = Polygon(poly_points)
return poly.contains(Point(pt))


def bbox_intersects_roi(bbox: Tuple[float,float,float,float], roi) -> bool:
# bbox = (x1,y1,x2,y2)
b = box(bbox[0], bbox[1], bbox[2], bbox[3])
if roi is None:
return True
if roi.get('type') == 'polygon':
poly = Polygon(roi.get('points', []))
return b.intersects(poly)
elif roi.get('type') == 'rect':
pts = roi.get('points', [])
if len(pts) >= 2:
x1,y1 = pts[0]
x2,y2 = pts[1]
r = box(x1,y1,x2,y2)
return b.intersects(r)
return False