import psycopg2

conn = psycopg2.connect(dbname='ecg_analysis', user='ecg_user', password='sais', host='127.0.0.1')
cur = conn.cursor()

cur.execute("SELECT MIN(segment_id), MAX(segment_id), COUNT(*) FROM ecg_features_annotatable WHERE dataset_source = 'Synthetic'")
min_id, max_id, count = cur.fetchone()

print(f"Synthetic Segments: {count}")
print(f"Range: {min_id} - {max_id}")

# Get a few specific IDs for PAC and PVC
cur.execute("SELECT segment_id FROM ecg_features_annotatable WHERE dataset_source = 'Synthetic' AND arrhythmia_label = 'PAC' LIMIT 3")
pacs = [r[0] for r in cur.fetchall()]
print(f"Sample PAC IDs: {pacs}")

cur.execute("SELECT segment_id FROM ecg_features_annotatable WHERE dataset_source = 'Synthetic' AND arrhythmia_label = 'PVC' LIMIT 3")
pvcs = [r[0] for r in cur.fetchall()]
print(f"Sample PVC IDs: {pvcs}")

conn.close()
