# app/models.py (UPDATED)
from sqlalchemy import Table, Column, Integer, String, Text, JSON, Float, Boolean, TIMESTAMP, MetaData, DateTime, BigInteger


metadata = MetaData()


atcc_cameras = Table(
    'atcc_cameras', metadata,
    Column('camera_id', Integer, primary_key=True, autoincrement=True),
    Column('camera_name', String(128), nullable=False),
    Column('rtsp_url', Text, nullable=False),
    Column('location', String(255)),
    Column('roi', JSON, nullable=True),
    Column('fps', Integer, default=15),
    Column('enabled', Boolean, default=True),
    # Note: TIMESTAMP type in SQLAlchemy defaults to CURRENT_TIMESTAMP on creation/update 
    # depending on the dialect. We leave the explicit default setting to the CREATE TABLE DDL.
    Column('last_seen', TIMESTAMP, nullable=True), 
)


atcc_records = Table(
    'atcc_records', metadata,
    Column('id', BigInteger, primary_key=True, autoincrement=True),
    Column('detection_id', String(36), nullable=False),
    Column('camera_id', Integer, nullable=False),
    
    # === NEW COLUMNS ADDED ===
    # Matches the new schema definition in your database
    Column('camera_name', String(255), default=None),
    Column('is_sync', Boolean, default=False),
    Column('sync_status', String(255), default='pending'),
    # =========================
    
    Column('detected_class', String(64)),
    Column('confidence', Float),
    Column('bbox', JSON),
    Column('centroid', JSON),
    Column('roi_hit', Boolean, default=False),
    Column('image_path', String(512)),
    # Assuming DATETIME(6) handles the passage time correctly
    Column('passage_time', DateTime),
    Column('inference_ms', Integer),
    Column('extra', JSON),
)