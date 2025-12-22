import numpy as np
try:
    import psycopg2
except ImportError:
    psycopg2 = None
import json
import random
import uuid

# Configuration
FS = 360  # Sampling rate (Hz) matches MIT-BIH usually
DURATION = 10  # Seconds
DB_PARAMS = {
    "dbname": "ecg_analysis",
    "user": "ecg_user",
    "password": "sais",
    "host": "127.0.0.1",
    "port": "5432"
}

class SyntheticECGGenerator:
    def __init__(self, fs=360, duration=10):
        self.fs = fs
        self.duration = duration
        self.length = int(fs * duration)
        
        # --- MORPHOLOGY TEMPLATES ---
        self.templates = {
            'NSR': { # Normal P-QRS-T
                'P': (0.15, -0.2, 0.02),
                'Q': (-0.15, -0.05, 0.01),
                'R': (1.0, 0.0, 0.01),
                'S': (-0.25, 0.05, 0.01),
                'T': (0.3, 0.3, 0.06)
            },
            'PAC': { # Early, weird P-wave
                'P': (0.2, -0.15, 0.015), 
                'Q': (-0.15, -0.05, 0.01),
                'R': (1.0, 0.0, 0.01),
                'S': (-0.25, 0.05, 0.01),
                'T': (0.3, 0.3, 0.06)
            },
            'PVC': { # Wide QRS, No P, Inverted T
                'R': (1.2, 0.0, 0.04),   # Wide
                'S': (-0.5, 0.1, 0.04),
                'T': (-0.4, 0.4, 0.08)
            },
            'Junctional': { # Inverted P or hidden P
                'P': (-0.1, -0.1, 0.015), 
                'Q': (-0.15, -0.05, 0.01),
                'R': (1.0, 0.0, 0.01),
                'S': (-0.25, 0.05, 0.01),
                'T': (0.3, 0.3, 0.06)
            },
            'LBBB': { # Wide QRS, Notched R
                'P': (0.15, -0.2, 0.02),
                'R': (0.8, -0.02, 0.03), # Notched R 1
                'R2': (0.8, 0.02, 0.03), # Notched R 2
                'T': (-0.2, 0.3, 0.08)   # Inverted T common
            },
            'RBBB': { # Wide S wave
                'P': (0.15, -0.2, 0.02),
                'R': (1.0, 0.0, 0.01),
                'S': (-0.4, 0.06, 0.03), # Slurred S
                'T': (0.3, 0.3, 0.06)
            }
        }

    def _gaussian(self, x, amp, center, width):
        return amp * np.exp(-((x - center)**2) / (2 * width**2))

    def generate_beat(self, beat_type='NSR'):
        t = np.linspace(-0.5, 0.5, int(self.fs))
        signal = np.zeros_like(t)
        
        template = self.templates.get(beat_type, self.templates['NSR'])
        
        for wave_name, params in template.items():
            if len(params) == 3:
                amp, center, width = params
                signal += self._gaussian(t, amp, center, width)
            
        return signal, int(self.fs / 2)

    def generate_segment(self, label, num_events=0):
        """
        Main logic for different rhythm types.
        Supports combined labels with '+' e.g. "Sinus Bradycardia + PVC"
        """
        signal = np.zeros(self.length)
        r_peaks = []
        
        # --- PARSE COMPOUND LABELS ---
        primary_label = label
        secondary_label = ""
        if " + " in label:
            parts = label.split(" + ")
            primary_label = parts[0]
            secondary_label = parts[1]
        elif "+" in label:
             parts = label.split("+")
             primary_label = parts[0].strip()
             secondary_label = parts[1].strip()

        # Combine logic: Check both
        full_check = primary_label + " " + secondary_label
        
        # --- 1. DETERMINE PARAMETERS BASED ON LABEL ---
        bpm = 70
        regularity = 'Regular' # Regular, Irregular, Irregularly Irregular
        beat_morphology = 'NSR'
        noise_level = 0.02
        
        # Baselines (Primary)
        if 'Bradycardia' in primary_label:
            bpm = random.uniform(40, 55)
        elif 'Tachycardia' in primary_label:
            bpm = random.uniform(110, 140)
            if 'Supra' in primary_label: # SVT
                bpm = random.uniform(150, 220)
                regularity = 'Strict'
        else:
            bpm = random.uniform(60, 90)

        if 'Fibrillation' in primary_label:
            regularity = 'Irregularly Irregular'
            if 'Atrial' in primary_label:
                beat_morphology = 'NSR' # QRS is narrow usually
            elif 'Ventricular' in primary_label:
                # VFib is special case - no beats
                pass

        if 'Flutter' in primary_label:
            regularity = 'Regular' # Usually fixed conduction
            bpm = 150 # 2:1 block common
        
        if 'Junctional' in primary_label:
             beat_morphology = 'Junctional'
        
        if 'Idioventricular' in primary_label:
             beat_morphology = 'PVC' # Wide QRS rhythm
             bpm = random.uniform(20, 40)
             
        if 'Paced' in primary_label:
             pass

        if 'Block' in primary_label:
             if 'Bundle Branch' in primary_label:
                 beat_morphology = 'LBBB' if 'Left' in primary_label else 'RBBB'
                 if 'Left' not in primary_label and 'Right' not in primary_label: beat_morphology = 'LBBB'
        
        # --- 2. GENERATE BEAT SCHEDULE (TIMING) ---
        
        beat_locs = []
        beat_types = []
        
        # Special Case: VFib
        if primary_label == 'Ventricular Fibrillation':
            t_full = np.arange(self.length) / self.fs
            for _ in range(5):
                f = random.uniform(3, 8)
                a = random.uniform(0.1, 0.5)
                ph = random.uniform(0, 2*np.pi)
                signal += a * np.sin(2*np.pi * f * t_full + ph)
            return signal, [] 

        # Timeline generation
        current_sample = int(0.2 * self.fs)
        
        # State variables for AV Blocks
        wenckebach_pr = 0.16 
        
        while current_sample < self.length - int(0.4*self.fs):
            
            # Determine Next Interval
            rr_s = 60.0 / bpm
            
            if regularity == 'Irregularly Irregular': # AFib
                rr_s = rr_s * random.uniform(0.6, 1.4)
            elif regularity == 'Irregular':
                rr_s = rr_s + random.uniform(-0.05, 0.05)
            else:
                rr_s = rr_s # Strict
            
            # Handle AV Block Drops
            is_dropped = False
            
            if "Type 1 Wenckebach" in primary_label:
                wenckebach_pr += 0.04
                if wenckebach_pr > 0.32:
                    is_dropped = True 
                    wenckebach_pr = 0.16 
                    
            elif "Type 2 Mobitz II" in primary_label:
                if random.random() < 0.25: 
                    is_dropped = True
            
            elif "3' AV Block" in primary_label:
                 pass 

            if not is_dropped:
                beat_locs.append(current_sample)
                beat_types.append(beat_morphology)
            
            current_sample += int(rr_s * self.fs)

        # --- 3. APPLY ECTOPIC PATTERNS (Secondary or Primary) ---
        
        # Check both primary and secondary for ectopic keywords
        check_labels = [primary_label, secondary_label]
        
        for lbl in check_labels:
            if not lbl: continue
            
            if 'Bigeminy' in lbl:
                ectopic_type = 'PVC' if 'PVC' in lbl or 'Ventricular' in lbl else 'PAC'
                for i in range(1, len(beat_types), 2):
                    beat_types[i] = ectopic_type
                    dist_prev = beat_locs[i] - beat_locs[i-1]
                    beat_locs[i] = beat_locs[i-1] + int(dist_prev * 0.6)
                    
            elif 'Trigeminy' in lbl:
                ectopic_type = 'PVC' if 'PVC' in lbl or 'Ventricular' in lbl else 'PAC'
                for i in range(2, len(beat_types), 3):
                    beat_types[i] = ectopic_type
                    dist_prev = beat_locs[i] - beat_locs[i-1]
                    beat_locs[i] = beat_locs[i-1] + int(dist_prev * 0.6)
            
            elif 'Couplet' in lbl or 'Pair' in lbl:
                 ectopic_type = 'PVC' if 'PVC' in lbl or 'Ventricular' in lbl else 'PAC'
                 if len(beat_types) > 4:
                     idx = random.randint(1, len(beat_types)-3)
                     beat_types[idx] = ectopic_type
                     beat_types[idx+1] = ectopic_type
                     beat_locs[idx] = beat_locs[idx-1] + int((beat_locs[idx]-beat_locs[idx-1])*0.6)
                     beat_locs[idx+1] = beat_locs[idx] + int((beat_locs[idx+1]-beat_locs[idx])*0.5)
                     
            elif ('PAC' in lbl or 'PVC' in lbl) and 'Bigeminy' not in lbl and 'Trigeminy' not in lbl and 'Couplet' not in lbl:
                # Simple random ectopics
                ectopic_type = 'PVC' if 'PVC' in lbl else 'PAC'
                # Default to 2 events if simple label, otherwise use passed num_events
                count_loc = num_events if num_events > 0 else 2
                
                # If we are adding PVCs to a baseline that isn't ectopic, execute
                for _ in range(count_loc):
                    if len(beat_types) > 4:
                        idx = random.randint(1, len(beat_types)-2)
                        # Don't overwrite if already ectopic (from Bigeminy etc)
                        if beat_types[idx] == beat_morphology: 
                            beat_types[idx] = ectopic_type
                            prev_dist = beat_locs[idx] - beat_locs[idx-1]
                            beat_locs[idx] = beat_locs[idx-1] + int(prev_dist * 0.6)

        # --- 4. RENDER SIGNAL ---
        
        for i, loc in enumerate(beat_locs):
            b_type = beat_types[i]
            beat_sig, center_offset = self.generate_beat(b_type)
            
            # 1ST DEGREE AV BLOCK MODIFIER
            if "1' AV Block" in primary_label and b_type == 'NSR':
                 pass 

            # RENDER
            start_idx = loc - center_offset
            end_idx = start_idx + len(beat_sig)
            
            if start_idx < 0:
                beat_sig = beat_sig[-start_idx:]
                start_idx = 0
            if end_idx > self.length:
                beat_sig = beat_sig[:-(end_idx - self.length)]
                end_idx = self.length
            
            signal[start_idx:end_idx] += beat_sig
            r_peaks.append(loc)

        # --- 5. ADD BACKGROUND FEATURES ---
        
        t_full = np.arange(self.length) / self.fs
        
        if 'Atrial Fibrillation' in primary_label:
            f_waves = 0.05 * np.sin(2 * np.pi * 6 * t_full) + 0.03 * np.sin(2*np.pi*9*t_full)
            signal += f_waves
        
        if 'Atrial Flutter' in primary_label:
            sawtooth = 0.15 * (t_full * 5 - np.floor(t_full * 5 + 0.5))
            signal += sawtooth

        signal += np.random.normal(0, 0.02, self.length)

        return signal, r_peaks

def save_to_db(generator, labels_to_gen):
    if psycopg2 is None:
        print("⚠️  Error: 'psycopg2' module not found. Cannot save to database.")
        print("   Run: pip install psycopg2-binary")
        return

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        for label, count in labels_to_gen.items():
            if count == 0: continue
            print(f"Generating {count} of {label}...")
            for i in range(count):
                signal, r_peaks = generator.generate_segment(label, num_events=2)
                
                filename = f"syn_{label.replace(' ','_').replace('+','_')[:15]}_{uuid.uuid4().hex[:6]}"
                
                cur.execute("""
                    INSERT INTO ecg_features_annotatable 
                    (filename, segment_index, segment_start_s, segment_duration_s, 
                     arrhythmia_label, raw_signal, r_peaks_in_segment, segment_fs, dataset_source,
                     features_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    filename, 0, 0.0, 10.0, label, 
                    signal.tolist(), json.dumps(r_peaks), 360, "Synthetic_Combos",
                    json.dumps({"label": label})
                ))
        
        conn.commit()
        cur.close()
        conn.close()
        print("Success!")
    except Exception as e:
        print(f"DB Error: {e}")

if __name__ == "__main__":
    gen = SyntheticECGGenerator()
    
    # COMPREHENSIVE LIST WITH COMBINATIONS
    worklist = {
        # Baseline (0 if already generated enough)
        "Sinus Rhythm": 0,
        
        # New Combinations
        "Sinus Bradycardia + PVC": 30,
        "Sinus Tachycardia + PVC": 30,
        "Sinus Bradycardia + PAC": 30,
        "Sinus Tachycardia + PAC": 30,
        
        "Atrial Fibrillation + PVC": 30,
        "Atrial Flutter + PVC": 30,
        "1st Degree AV Block + PVC": 30,
        
        # Complex Combos
        "Sinus Bradycardia + PVC Bigeminy": 20,
        "Sinus Tachycardia + PVC Bigeminy": 20,
    }
    
    save_to_db(gen, worklist)
