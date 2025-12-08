import tkinter as tk
import customtkinter as ctk
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import wfdb
import threading
import queue
import time
import sys

# --- CONFIGURATION ---
try:
    from synthesize_arrhythmias import SyntheticECGGenerator
    HAS_SYNTH = True
except ImportError:
    HAS_SYNTH = False
    print("Warning: ensure synthesize_arrhythmias.py is available for advanced rhythms.")

SAMPLE_RATE = 48000     # Audio DAC Sample Rate
BLOCK_SIZE = 2048       # Audio Buffer Size
MIT_RESAMPLE_RATE = 360 # Native MIT-BIH Hz

# Visualization Settings
VISUAL_DOWNSAMPLE = 80  # Lower downsample for smoother lines
WINDOW_SECONDS = 10     # Standard ECG strip length
PLOT_POINTS = int((SAMPLE_RATE / VISUAL_DOWNSAMPLE) * WINDOW_SECONDS)

# Set Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Curated List of MIT-BIH Records
MIT_RECORD_MAP = {
    "100: Normal Sinus Rhythm": "100",
    "101: Normal Sinus Rhythm (Ex 2)": "101",
    "106: PVCs (Ventricular Ectopy)": "106",
    "109: LBBB": "109",
    "118: RBBB": "118",
    "119: Ventricular Bigeminy": "119",
    "200: Ventricular Bigeminy (2)": "200",
    "201: Atrial Fibrillation": "201",
    "203: Multi-modal Arrhythmia": "203",
    "205: Ventricular Tachycardia": "205",
    "230: WPW Syndrome": "230",
    "231: RBBB & 1st Deg Block": "231"
}

class SignalEngine:
    def __init__(self):
        self.mode = "Synthetic" 
        self.waveform = "NSR"
        self.target_mv = 1.0    # Desired Output in mV
        self.calibration_gain = 0.5 # The DAC float value that equals 1mV (set by user)
        self.is_calibrated = False
        
        self.bpm = 60
        self.fs = SAMPLE_RATE
        self.phase = 0
        
        # Synthetic ECG Params (Lead II approx)
        self.ecg_params = {
            'P': (0.15, -0.2, 0.02), 'Q': (-0.15, -0.05, 0.01),
            'R': (1.0, 0.0, 0.01), 'S': (-0.25, 0.05, 0.01),
            'T': (0.3, 0.3, 0.06)
        }
        
        # Advanced Generator (High Res for Audio)
        self.complex_gen = None
        self.complex_buffer = None
        self.complex_idx = 0
        
        if HAS_SYNTH:
            # We use a lower FS for generation to keep it fast, then linear interp if needed, 
            # OR just generate at high res. 48kHz is fine for 10s (480k floats ~ 4MB RAM)
            self.complex_gen = SyntheticECGGenerator(fs=SAMPLE_RATE, duration=10)
        
        self.mit_data = None
        self.mit_index = 0
        self.plot_queue = queue.Queue(maxsize=100)

    def _gaussian(self, t, amp, center, width):
        return amp * np.exp(-((t - center)**2) / (2 * width**2))

    def get_output_gain(self):
        """Calculates final DAC amplitude (0.0-1.0) based on target mV and calibration."""
        if not self.is_calibrated:
            # If not calibrated, target_mv acts as raw 0-1 percentage (scaled 0-5 for UI feel)
            return np.clip(self.target_mv / 5.0, 0, 1.0) 
        else:
            # If calibrated: 1mV = calibration_gain. 
            # So X mV = X * calibration_gain
            return np.clip(self.target_mv * self.calibration_gain, 0, 1.0)

    def load_mit_record(self, record_id, progress_callback=None):
        try:
            if progress_callback: progress_callback(0.1)
            # pn_dir='mitdb' fetches from PhysioNet
            record = wfdb.rdrecord(record_id, pn_dir='mitdb', channels=[0])
            if progress_callback: progress_callback(0.5)
            
            signal = record.p_signal.flatten()
            
            # Normalize to -1.0 to 1.0 range
            max_val = np.max(np.abs(signal))
            if max_val > 0: signal = signal / max_val
            
            # Linear Resample
            original_len = len(signal)
            duration_sec = original_len / MIT_RESAMPLE_RATE
            target_len = int(duration_sec * self.fs)
            
            if progress_callback: progress_callback(0.7)
            
            self.mit_data = np.interp(
                np.linspace(0, original_len, target_len),
                np.arange(original_len),
                signal
            )
            self.mit_index = 0
            
            if progress_callback: progress_callback(1.0)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def audio_callback(self, outdata, frames, time_info, status):
        buffer = np.zeros(frames)
        
        # --- FREQUENCY LOGIC UPDATE ---
        # For Test Signals: Slider Value = Hz
        # For ECG Signals: Slider Value = BPM (so freq = BPM/60)
        
        if self.waveform in ["NSR", "PVC"]:
            freq = self.bpm / 60.0 # Standard Cardiac Rhythm
        else:
            freq = self.bpm        # Direct Hz mapping (60 slider = 60Hz)

        if self.mode == "Calibration":
             # Fixed 1Hz Square Wave for Calibration
            t = (np.arange(frames) + self.phase) / self.fs
            buffer = np.sign(np.sin(2 * np.pi * 1.0 * t))
            self.phase += frames

        elif self.mode == "Synthetic":
            t = (np.arange(frames) + self.phase) / self.fs
            
            if self.waveform == "Sine":
                buffer = np.sin(2 * np.pi * freq * t)
            
            elif self.waveform == "Square":
                buffer = np.sign(np.sin(2 * np.pi * freq * t))
            
            elif self.waveform == "Sawtooth":
                # 2 * (t * f - floor(t*f + 0.5)) for centered sawtooth
                buffer = 2 * (t * freq - np.floor(t * freq + 0.5))
                
            elif self.waveform in ["NSR", "PVC"]:
                # Legacy simple synthesis (kept for simple logic)
                period_samples = int(self.fs / freq)
                if period_samples == 0: period_samples = 1 
                
                local_t = np.arange(frames) + self.phase
                cycle_pos = (local_t % period_samples) / period_samples
                cycle_t = cycle_pos - 0.5 
                
                is_pvc = (self.waveform == "PVC")
                if is_pvc:
                    p_list = [(0.0, -0.2, 0.02), (1.2, 0.0, 0.04), (-0.5, 0.1, 0.04), (-0.4, 0.4, 0.08)]
                else:
                    p_list = self.ecg_params.values()
                
                for a, c, w in p_list:
                    buffer += a * np.exp(-((cycle_t - c)**2) / (2 * w**2))

            else:
                # Comples Rhythms (Junctional, Blocks, AFib, etc.)
                if self.complex_buffer is None:
                     # Lazy generation
                     if self.complex_gen:
                         print(f"Generating {self.waveform}...")
                         # remap name if needed
                         # synthesize_arrhythmias expects specific strings
                         sig, _ = self.complex_gen.generate_segment(self.waveform)
                         # Normalize
                         if np.max(np.abs(sig)) > 0: sig /= np.max(np.abs(sig))
                         self.complex_buffer = sig
                         self.complex_idx = 0
                
                if self.complex_buffer is not None:
                     # Ring buffer playback
                     remaining = len(self.complex_buffer) - self.complex_idx
                     if remaining >= frames:
                         buffer[:] = self.complex_buffer[self.complex_idx : self.complex_idx+frames]
                         self.complex_idx += frames
                     else:
                         buffer[:remaining] = self.complex_buffer[self.complex_idx:]
                         buffer[remaining:] = self.complex_buffer[:frames-remaining]
                         self.complex_idx = frames - remaining

            self.phase += frames

        elif self.mode == "MIT-BIH":
            if self.mit_data is None:
                buffer[:] = 0
            else:
                remaining = len(self.mit_data) - self.mit_index
                if remaining >= frames:
                    buffer[:] = self.mit_data[self.mit_index:self.mit_index+frames]
                    self.mit_index += frames
                else:
                    buffer[:remaining] = self.mit_data[self.mit_index:]
                    start_len = frames - remaining
                    buffer[remaining:] = self.mit_data[:start_len]
                    self.mit_index = start_len

        # Apply Gain
        # If in calibration mode, use raw target_mv (0-1) as gain
        if self.mode == "Calibration":
            gain = self.target_mv # Raw slider value 0-1
        else:
            gain = self.get_output_gain()
            
        final_output = buffer * gain
        outdata[:] = final_output.reshape(-1, 1)
        
        try:
            self.plot_queue.put_nowait(final_output[::VISUAL_DOWNSAMPLE]) 
        except queue.Full:
            pass

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Biomedical Signal Generator - PhD Workstation")
        self.geometry("1200x800")
        
        self.engine = SignalEngine()
        self.stream = None
        self.running = True # Flag to control loops
        
        self._setup_ui()
        self._start_audio_stream()
        self._start_scan_plot()

    def _setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(12, weight=1)

        # Header
        ctk.CTkLabel(self.sidebar, text="ECG SIMULATOR", font=ctk.CTkFont(size=24, weight="bold"), text_color="#2CC985").grid(row=0, column=0, padx=20, pady=(20, 5))
        ctk.CTkLabel(self.sidebar, text="v1.0 Release", font=ctk.CTkFont(size=10)).grid(row=1, column=0, padx=20, pady=(0, 20))

        # 1. Mode Switch
        self.mode_label = ctk.CTkLabel(self.sidebar, text="OPERATION MODE", anchor="w", font=ctk.CTkFont(size=12, weight="bold"))
        self.mode_label.grid(row=2, column=0, padx=20, sticky="w")
        self.mode_seg = ctk.CTkSegmentedButton(self.sidebar, values=["Synthetic", "MIT-BIH"], command=self.change_mode)
        self.mode_seg.set("Synthetic")
        self.mode_seg.grid(row=3, column=0, padx=20, pady=5)

        # 2. Waveform Selector
        # Comprehensive List
        wave_options = [
            # Standard
            "NSR", "Sinus Bradycardia", "Sinus Tachycardia", 
            "Atrial Fibrillation", "Atrial Flutter", "Junctional Rhythm",
            "PVC", "PVC Bigeminy", "PVC Trigeminy", "PVC Couplet",
            "PAC", "Ventricular Tachycardia", "Ventricular Fibrillation",
            "Idioventricular Rhythm", "Bundle Branch Block",
            "2' AV Block, Type 1 Wenckebach", "2' AV Block, Type 2 Mobitz II",
            # Combinations
            "Sinus Bradycardia + PVC", "Sinus Tachycardia + PVC",
            "Sinus Bradycardia + PAC", "Sinus Tachycardia + PAC",
            "Atrial Fibrillation + PVC", "Atrial Flutter + PVC",
            "1st Degree AV Block + PVC", "Sinus Bradycardia + PVC Bigeminy",
            # Test
            "Sine", "Square"
        ]
        self.wave_opt = ctk.CTkOptionMenu(self.sidebar, values=wave_options, command=self.change_waveform)
        self.wave_opt.grid(row=4, column=0, padx=20, pady=10)

        # 3. Frequency / BPM
        self.freq_label_title = ctk.CTkLabel(self.sidebar, text="FREQUENCY / RATE", anchor="w", font=ctk.CTkFont(size=12, weight="bold"))
        self.freq_label_title.grid(row=5, column=0, padx=20, pady=(10,0), sticky="w")
        
        bpm_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bpm_frame.grid(row=6, column=0, padx=20, pady=0, sticky="ew")
        
        self.bpm_slider = ctk.CTkSlider(self.sidebar, from_=10, to=180, number_of_steps=170, command=self.update_bpm_slider)
        self.bpm_slider.set(60)
        self.bpm_slider.grid(row=7, column=0, padx=20, pady=(0, 10))
        
        self.bpm_val_label = ctk.CTkLabel(bpm_frame, text="60 BPM")
        self.bpm_val_label.pack(side="right")

        # 4. Amplitude & Calibration
        ctk.CTkLabel(self.sidebar, text="SIGNAL AMPLITUDE", anchor="w", font=ctk.CTkFont(size=12, weight="bold")).grid(row=8, column=0, padx=20, pady=(10,0), sticky="w")
        
        amp_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        amp_frame.grid(row=9, column=0, padx=20, pady=0, sticky="ew")
        self.amp_val_label = ctk.CTkLabel(amp_frame, text="Gain: 50%")
        self.amp_val_label.pack(side="right")
        
        self.amp_slider = ctk.CTkSlider(self.sidebar, from_=0, to=5, number_of_steps=100, command=self.update_amp_slider)
        self.amp_slider.set(1.0) # Default mid range
        self.amp_slider.grid(row=10, column=0, padx=20, pady=(0, 10))

        self.calib_btn = ctk.CTkButton(self.sidebar, text="Step 1: Calibrate Output", fg_color="#444", command=self.toggle_calibration_mode)
        self.calib_btn.grid(row=11, column=0, padx=20, pady=5)

        # 5. MIT Loader
        mit_frame = ctk.CTkFrame(self.sidebar, fg_color="#2b2b2b")
        mit_frame.grid(row=13, column=0, padx=10, pady=20, sticky="ew")
        
        ctk.CTkLabel(mit_frame, text="MIT-BIH DATABASE", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)
        self.mit_opt = ctk.CTkOptionMenu(mit_frame, values=list(MIT_RECORD_MAP.keys()))
        self.mit_opt.pack(pady=5, padx=10, fill="x")
        
        self.mit_progress = ctk.CTkProgressBar(mit_frame, width=200)
        self.mit_progress.set(0)
        self.mit_progress.pack(pady=5, padx=10)
        
        self.load_btn = ctk.CTkButton(mit_frame, text="Download Record", command=self.load_mit_data)
        self.load_btn.pack(pady=10, padx=10)

        # --- Main Monitor Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#000000")
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        
        # Monitor Header
        header = ctk.CTkFrame(self.main_frame, height=40, fg_color="#111")
        header.pack(fill="x", side="top")
        ctk.CTkLabel(header, text="  LEAD II  ", font=ctk.CTkFont(family="Consolas", weight="bold"), text_color="#00ff00").pack(side="left")
        self.status_ind = ctk.CTkLabel(header, text="● LIVE  ", text_color="red")
        self.status_ind.pack(side="right")

        # Matplotlib
        self.fig, self.ax = plt.subplots(facecolor='black')
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        self.ax.set_facecolor('black')
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_xlim(0, PLOT_POINTS)
        
        # Grid settings mimicking patient monitor
        self.ax.grid(True, color='#222', linestyle='-', linewidth=0.5)
        self.ax.set_xticks(np.arange(0, PLOT_POINTS, PLOT_POINTS/10)) # Vertical grid lines
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # The Line
        self.line, = self.ax.plot(np.zeros(PLOT_POINTS), color='#00ff00', linewidth=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- Callbacks ---
    def update_bpm_slider(self, value):
        self.engine.bpm = value
        
        # Smart Labeling Logic
        if self.engine.waveform in ["NSR", "PVC"]:
            # Standard Cardiac (BPM)
            freq = value / 60.0
            self.bpm_val_label.configure(text=f"{int(value)} BPM")
        else:
            # Test Signals (Hz)
            self.bpm_val_label.configure(text=f"{int(value)} Hz")

    def update_amp_slider(self, value):
        self.engine.target_mv = value
        if self.engine.is_calibrated:
            self.amp_val_label.configure(text=f"Output: {value:.2f} mV")
        else:
            # In Uncalibrated mode, slider is 0-5, we treat it as 0-100% gain roughly
            pct = (value / 5.0) * 100
            self.amp_val_label.configure(text=f"Gain: {int(pct)}%")

    def change_mode(self, value):
        self.engine.mode = value
        if value == "MIT-BIH":
            self.wave_opt.configure(state="disabled")
            self.bpm_slider.configure(state="disabled")
        else:
            self.wave_opt.configure(state="normal")
            self.bpm_slider.configure(state="normal")

    def change_waveform(self, value):
        self.engine.waveform = value
        self.engine.complex_buffer = None # Force regeneration
        # Refresh the slider label to show the correct unit (BPM vs Hz) immediately
        self.update_bpm_slider(self.bpm_slider.get())

    def toggle_calibration_mode(self):
        if self.engine.mode == "Calibration":
            # Finish Calibration
            self.engine.mode = "Synthetic"
            self.engine.calibration_gain = self.amp_slider.get() # Save the raw slider val as the 1mV ref
            self.engine.is_calibrated = True
            
            # Reset UI
            self.calib_btn.configure(text="Recalibrate System", fg_color="#444")
            self.mode_label.configure(text="OPERATION MODE")
            self.amp_slider.configure(from_=0, to=5) # Now range is 0mV to 5mV
            self.amp_slider.set(1.0)
            self.engine.target_mv = 1.0
            self.amp_val_label.configure(text="Output: 1.00 mV")
            
            tk.messagebox.showinfo("Success", "System Calibrated.\nSlider now sets actual Millivolts (0-5mV).")
            
        else:
            # Start Calibration
            self.engine.mode = "Calibration"
            self.calib_btn.configure(text="CONFIRM 1mV ON SCOPE", fg_color="red")
            self.mode_label.configure(text="MODE: CALIBRATION (1Hz SQ)")
            
            # Set slider to raw DAC mode (0.0 to 1.0)
            self.amp_slider.configure(from_=0, to=1.0)
            self.amp_slider.set(0.1) # Start low for safety
            self.engine.target_mv = 0.1
            self.amp_val_label.configure(text="DAC Level (Raw)")
            
            tk.messagebox.showinfo("Calibration", 
                                   "1. Connect Scope/Monitor.\n"
                                   "2. Signal is now 1Hz Square Wave.\n"
                                   "3. Adjust Slider until Scope shows exactly 1mVpp.\n"
                                   "4. Click 'CONFIRM' button.")

    def load_mit_data(self):
        selection = self.mit_opt.get()
        rec_id = MIT_RECORD_MAP[selection]
        
        self.load_btn.configure(state="disabled", text="Downloading...")
        self.mit_progress.set(0)
        
        def update_prog(val):
            self.mit_progress.set(val)
        
        def _thread():
            success = self.engine.load_mit_record(rec_id, progress_callback=update_prog)
            if success:
                self.mode_seg.set("MIT-BIH")
                self.change_mode("MIT-BIH")
                self.load_btn.configure(text="Loaded OK", state="normal")
            else:
                self.load_btn.configure(text="Error", state="normal")
                
        threading.Thread(target=_thread, daemon=True).start()

    def _start_audio_stream(self):
        self.stream = sd.OutputStream(
            channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
            callback=self.engine.audio_callback
        )
        self.stream.start()

    def _start_scan_plot(self):
        # Buffer for plot data
        self.y_data = np.zeros(PLOT_POINTS)
        self.scan_idx = 0
        self._animate_scan()

    def _animate_scan(self):
        if not self.running: return # Stop if app is closing
        
        try:
            # Process all chunks in queue
            while not self.engine.plot_queue.empty():
                chunk = self.engine.plot_queue.get_nowait()
                chunk_len = len(chunk)
                
                # We need to write chunk into circular buffer at scan_idx
                if self.scan_idx + chunk_len < PLOT_POINTS:
                    self.y_data[self.scan_idx : self.scan_idx + chunk_len] = chunk
                    
                    # Create "Gap" / Wiper effect
                    gap_start = self.scan_idx + chunk_len
                    gap_end = min(gap_start + 50, PLOT_POINTS)
                    self.y_data[gap_start : gap_end] = np.nan
                    
                    self.scan_idx += chunk_len
                else:
                    # Wrapping around
                    space_left = PLOT_POINTS - self.scan_idx
                    self.y_data[self.scan_idx:] = chunk[:space_left]
                    remaining = chunk_len - space_left
                    self.y_data[:remaining] = chunk[space_left:]
                    
                    # Gap at start
                    self.y_data[remaining : remaining + 50] = np.nan
                    self.scan_idx = remaining

            self.line.set_ydata(self.y_data)
            self.canvas.draw_idle()
            
        except Exception:
            pass
        
        if self.running:
            self.after(20, self._animate_scan)

    def on_close(self):
        self.running = False # Stop animation loop
        
        # Stop Audio Stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        
        # Stop Matplotlib interactions
        plt.close('all')
        
        # Stop Tkinter Mainloop and Destroy
        self.quit() 
        self.destroy()
        sys.exit()

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()