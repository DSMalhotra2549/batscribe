"""Batscribe GUI: drop a folder, press Start, get an .srt next to every video.

Launch with .venv\\Scripts\\pythonw.exe batscribe_gui.py [folder-or-video ...]
"""
import ctypes
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from tkinterdnd2 import DND_FILES, TkinterDnD

import batscribe as bs

NOTHING_SELECTED = "Nothing selected yet."


class App:
    def __init__(self, root, initial_inputs):
        self.root = root
        self.events = queue.Queue()      # worker thread -> UI thread
        self.stop_flag = threading.Event()
        self.inputs, self.todo, self.durations, self.rows = [], [], {}, {}
        self.busy = False                # scanning or transcribing
        self.running = False             # transcribing
        self.current = ""

        root.title("Batscribe")
        root.geometry("680x580")
        root.minsize(480, 400)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", self.on_drop)

        pad = {"padx": 12, "pady": (10, 0)}
        drop = tk.Frame(root, bg="#eef3fb", highlightbackground="#9db4d8", highlightthickness=2)
        drop.pack(fill="x", **pad)
        tk.Label(drop, text="Drop a folder or videos here", bg="#eef3fb", fg="#2b4a75",
                 font=("Segoe UI", 13)).pack(pady=(14, 4))
        pick = tk.Frame(drop, bg="#eef3fb")
        pick.pack(pady=(0, 14))
        self.browse_btn = ttk.Button(pick, text="Choose folder…", command=self.browse)
        self.browse_btn.pack(side="left", padx=4)
        self.files_btn = ttk.Button(pick, text="Choose videos…", command=self.browse_files)
        self.files_btn.pack(side="left", padx=4)

        self.summary = ttk.Label(root, text=NOTHING_SELECTED)
        self.summary.pack(fill="x", **pad)

        list_frame = ttk.Frame(root)
        list_frame.pack(fill="both", expand=True, **pad)
        self.tree = ttk.Treeview(list_frame, columns=("status",), selectmode="browse")
        self.tree.heading("#0", text="Video  (double-click to play)", anchor="w")
        self.tree.heading("status", text="Status", anchor="w")
        self.tree.column("status", width=130, stretch=False)
        self.tree.tag_configure("done", foreground="#1e7b34")
        self.tree.tag_configure("failed", foreground="#b3261e")
        self.tree.tag_configure("active", foreground="#1a56b0")
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", self.open_selected)

        self.progress = ttk.Progressbar(root, maximum=100)
        self.progress.pack(fill="x", **pad)
        self.stats = ttk.Label(root, text="")
        self.stats.pack(fill="x", padx=12, pady=(4, 0))

        buttons = ttk.Frame(root)
        buttons.pack(fill="x", padx=12, pady=12)
        self.start_btn = ttk.Button(buttons, text="Start", command=self.start, state="disabled")
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(8, 0))
        self.new_btn = ttk.Button(buttons, text="New batch", command=self.new_batch)
        self.new_btn.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Show in folder", command=self.show_in_folder).pack(side="right")
        ttk.Button(buttons, text="Logs", command=self.show_logs).pack(side="right", padx=(0, 8))

        root.after(100, self.pump)
        if initial_inputs:
            self.load(initial_inputs)

    # ---- choosing input -------------------------------------------------
    def on_drop(self, event):
        if not self.busy:
            self.load(list(self.root.tk.splitlist(event.data)))

    def browse(self):
        folder = filedialog.askdirectory(title="Choose a folder of videos")
        if folder:
            self.load([folder])

    def browse_files(self):
        exts = " ".join("*" + e for e in sorted(bs.VIDEO_EXTS))
        files = filedialog.askopenfilenames(title="Choose videos", filetypes=[("Videos", exts), ("All files", "*.*")])
        if files:
            self.load(list(files))

    def new_batch(self):
        """Clear the window so another folder or set of videos can be chosen."""
        self.inputs, self.todo, self.durations = [], [], {}
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()
        self.summary.config(text=NOTHING_SELECTED)
        self.progress["value"] = 0
        self.stats.config(text="")
        self.set_busy(False)

    def load(self, inputs):
        """Scan in the background (ffprobe on many files takes a few seconds)."""
        self.inputs = [str(Path(p)) for p in inputs]
        self.set_busy(True)
        self.summary.config(text="Scanning…")
        self.tree.delete(*self.tree.get_children())
        self.rows.clear()

        def scan():
            videos = bs.collect(self.inputs)
            todo = [v for v in videos if not v.with_suffix(".srt").exists()]
            durations = {v: bs.video_duration(v) for v in todo}
            self.events.put(("scanned", {"videos": videos, "todo": todo, "durations": durations}))

        threading.Thread(target=scan, daemon=True).start()

    def on_scanned(self, videos, todo, durations):
        self.todo, self.durations = todo, durations
        base = Path(self.inputs[0]) if Path(self.inputs[0]).is_dir() else None
        for v in videos:
            label = str(v.relative_to(base)) if base and v.is_relative_to(base) else v.name
            done = v not in durations
            self.rows[v] = self.tree.insert("", "end", text=label, tags=("done",) if done else (),
                                            values=("✔ already done" if done else "waiting",))
        where = self.inputs[0] if len(self.inputs) == 1 else f"{len(self.inputs)} items"
        self.summary.config(text=f"{where}\n{len(videos)} videos · {len(videos) - len(todo)} already captioned · "
                                 f"{len(todo)} to do ({bs.fmt_dur(sum(durations.values()))} of audio)")
        self.progress["value"] = 0
        self.stats.config(text="" if todo else "Nothing to do — every video already has captions.")
        self.set_busy(False)

    # ---- running --------------------------------------------------------
    def start(self):
        self.stop_flag.clear()
        self.running = True
        self.set_busy(True)
        self.stop_btn.config(state="normal")
        self.progress["value"] = 0

        def work():
            try:
                bs.run_batch(self.todo, self.durations, bs.error_log_path(self.inputs[0]),
                             on_event=lambda kind, **info: self.events.put((kind, info)),
                             should_stop=self.stop_flag.is_set)
            except Exception as e:  # e.g. model/GPU failed to load
                self.events.put(("crashed", {"error": str(e)}))

        threading.Thread(target=work, daemon=True).start()

    def stop(self):
        self.stop_flag.set()
        self.stop_btn.config(state="disabled")
        self.stats.config(text="Stopping… (the current video will be redone next time)")

    def pump(self):
        """Apply worker events on the UI thread."""
        while not self.events.empty():
            kind, info = self.events.get()
            getattr(self, "on_" + kind)(**info)
        self.root.after(100, self.pump)

    def set_status(self, video, text, tag=()):
        self.tree.item(self.rows[video], values=(text,), tags=tag)

    def on_loading(self, model):
        self.stats.config(text=f"Loading model {model} on the GPU…")

    def on_start(self, n, total, video, duration):
        self.set_status(video, "▶ starting…", ("active",))
        self.tree.see(self.rows[video])
        self.current = f"Video {n} of {total}"
        if not self.stop_flag.is_set():
            self.stats.config(text=f"{self.current} · decoding audio…")

    def on_progress(self, n, video, file_pct, overall_pct, speed, eta):
        self.set_status(video, f"▶ {file_pct:.0f}%", ("active",))
        self.progress["value"] = overall_pct
        self.root.title(f"{overall_pct:.0f}% · Batscribe")
        if not self.stop_flag.is_set():
            self.stats.config(text=f"{self.current} · {overall_pct:.1f}% overall · "
                                   f"{speed:.1f}x realtime · about {bs.fmt_dur(eta)} left")

    def on_done(self, n, video, srt):
        self.set_status(video, "✔ done", ("done",))

    def on_failed(self, n, video, error):
        self.set_status(video, "✖ failed", ("failed",))

    def on_finished(self, ok, failed, stopped, elapsed, log_path):
        for v in self.todo:  # the video that was interrupted, if any
            if self.tree.set(self.rows[v], "status").startswith("▶"):
                self.set_status(v, "stopped")
        self.todo = [v for v in self.todo if not v.with_suffix(".srt").exists()]
        self.running = False
        self.set_busy(False)
        self.root.title("Batscribe")
        msg = f"{'Stopped' if stopped else 'Finished'} after {bs.fmt_dur(elapsed)}: {ok} captioned, {failed} failed."
        self.stats.config(text=msg)
        if failed:
            msg += f"\n\nDetails of failures:\n{log_path}"
        if self.todo and not stopped:
            msg += "\n\nPress Start to retry the failed videos."
        self.notify("Batscribe", msg, error=bool(failed))

    def on_crashed(self, error):
        self.running = False
        self.set_busy(False)
        self.root.title("Batscribe")
        self.stats.config(text="Could not run.")
        self.notify("Batscribe – error", f"Transcription could not start or stopped unexpectedly:\n\n{error}", error=True)

    # ---- helpers --------------------------------------------------------
    def notify(self, title, msg, error=False):
        winsound.MessageBeep(winsound.MB_ICONHAND if error else winsound.MB_ICONASTERISK)
        self.root.deiconify()
        self.root.lift()
        (messagebox.showwarning if error else messagebox.showinfo)(title, msg, parent=self.root)

    def set_busy(self, busy):
        self.busy = busy
        for btn in (self.browse_btn, self.files_btn, self.new_btn):
            btn.config(state="disabled" if busy else "normal")
        self.start_btn.config(state="disabled" if busy or not self.todo else "normal")
        if not busy:
            self.stop_btn.config(state="disabled")

    def selected_video(self):
        sel = self.tree.selection()
        return next((v for v, item in self.rows.items() if sel and item == sel[0]), None)

    def open_selected(self, _event=None):
        video = self.selected_video()
        if video:
            os.startfile(video)  # players like VLC/MPC pick up the matching .srt automatically

    def show_in_folder(self):
        video = self.selected_video()
        target = video if video else (Path(self.inputs[0]) if self.inputs else None)
        if target:
            subprocess.Popen(f'explorer /select,"{target}"')

    def show_logs(self):
        if bs.HISTORY_LOG.exists():
            os.startfile(bs.HISTORY_LOG)  # opens in Notepad (or the default .log viewer)
        else:
            messagebox.showinfo("Batscribe", "Nothing has been converted yet.", parent=self.root)

    def on_close(self):
        if self.running:
            if not messagebox.askyesno("Batscribe", "Captioning is still running. Quit anyway?\n\n"
                                       "Finished videos are kept; the current one will be redone next time.",
                                       parent=self.root):
                return
            os._exit(0)  # don't wait for the GPU thread
        self.root.destroy()


def main():
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on high-DPI screens
    root = TkinterDnD.Tk()
    App(root, sys.argv[1:])
    root.mainloop()


if __name__ == "__main__":
    main()
