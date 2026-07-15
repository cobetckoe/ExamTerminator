import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
import threading
import queue
import json
import os
import sys
import time
import re
import subprocess
from difflib import SequenceMatcher, get_close_matches
import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from collections import Counter

CONFIG_FILE = "answer_config.json"

# ---------- ADB ----------
def find_adb():
    # exe 运行时，从 exe 所在目录或临时解压目录找
    if getattr(sys, 'frozen', False):
        bases = [os.path.dirname(sys.executable), sys._MEIPASS]
    else:
        bases = [os.path.dirname(os.path.abspath(__file__))]
    possible_paths = []
    for base in bases:
        possible_paths.extend([
            os.path.join(base, "adb.exe"),
            os.path.join(base, "platform-tools", "adb.exe"),
            os.path.join(base, "src", "platform-tools", "adb.exe"),
        ])
    possible_paths.extend(["adb.exe", "platform-tools\\adb.exe"])
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def restart_adb():
    """重启ADB服务"""
    if not ADB_PATH:
        return False
    try:
        subprocess.run([ADB_PATH, "kill-server"], capture_output=True, timeout=5)
        subprocess.run([ADB_PATH, "start-server"], capture_output=True, timeout=10)
        return True
    except:
        return False

def _to_manual(app):
    """统一异常处理：转手动模式"""
    app.log("发现异常，转手动模式")
    app.mode = "manual"
    app.mode_var.set("manual")
    app.last_question_key = ""

ADB_PATH = find_adb()
if not ADB_PATH:
    print("【严重错误】未找到 adb.exe！")

# 延迟初始化，不阻塞启动
PHONE_W, PHONE_H = 1080, 2400
AVAILABLE_CAMERAS = []
cap = None
_adb_inited = False
_cam_inited = False
_app_ref = None

def _init_adb():
    """后台初始化ADB（获取手机分辨率）"""
    global PHONE_W, PHONE_H, _adb_inited
    if not ADB_PATH:
        _adb_inited = True
        return
    # 重试3次获取分辨率
    for attempt in range(3):
        try:
            res = subprocess.run([ADB_PATH, "shell", "wm", "size"], capture_output=True, text=True, timeout=10)
            match = re.search(r'(\d+)x(\d+)', res.stdout)
            if match:
                PHONE_W, PHONE_H = int(match.group(1)), int(match.group(2))
                break
        except:
            if attempt < 2:
                time.sleep(1)
                # 重启ADB服务
                try:
                    subprocess.run([ADB_PATH, "kill-server"], capture_output=True, timeout=5)
                    subprocess.run([ADB_PATH, "start-server"], capture_output=True, timeout=10)
                except:
                    pass
    _adb_inited = True

def _init_cameras():
    """后台初始化摄像头检测"""
    global AVAILABLE_CAMERAS, _cam_inited
    for idx in range(2):
        try:
            try:
                old_level = cv2.getLogLevel()
                cv2.setLogLevel(0)
            except:
                old_level = None
            test_cap = cv2.VideoCapture(idx)
            if old_level is not None:
                try: cv2.setLogLevel(old_level)
                except: pass
            if test_cap.isOpened():
                # 只检查是否能打开，不读帧（读帧太慢）
                test_cap.release()
                AVAILABLE_CAMERAS.append(idx)
        except:
            pass
    _cam_inited = True
    if hasattr(_app_ref, 'root'):
        _app_ref.root.after(0, _app_ref._refresh_cameras)

def capture_frame():
    """截图返回numpy帧，失败返回None"""
    global cap
    if cap is None or not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 5:
        return None
    return frame

def ocr_frame(ocr_engine, frame):
    """对numpy帧执行OCR，返回结果列表"""
    if frame is None:
        return None
    result, _ = ocr_engine(frame)
    return result

def ocr_with_consensus(ocr_engine, times=1):
    """截图OCR，times>1时多次取众数（默认1次即可，省时间）"""
    frames = []
    for _ in range(max(1, times)):
        f = capture_frame()
        if f is not None:
            frames.append(f)
        if times > 1:
            time.sleep(0.1)

    if not frames:
        return None

    all_results = []
    for f in frames:
        r = ocr_frame(ocr_engine, f)
        if r:
            all_results.append(r)

    if not all_results:
        return None
    if len(all_results) == 1:
        return all_results[0]

    # 众数合并
    base = all_results[0]
    consensus = []
    for base_line in base:
        base_box = base_line[0]
        base_text = base_line[1]
        base_score = base_line[2] if len(base_line) > 2 else 1.0
        base_y = min(pt[1] for pt in base_box)
        texts = [base_text]
        scores = [base_score]
        for other_result in all_results[1:]:
            best_match = None
            best_dist = 999
            for other_line in other_result:
                other_y = min(pt[1] for pt in other_line[0])
                dist = abs(other_y - base_y)
                text_match = (other_line[1] == base_text)
                if (dist < 50 and dist < best_dist) or (text_match and dist < 200):
                    best_dist = dist
                    best_match = other_line
            if best_match:
                texts.append(best_match[1])
                scores.append(best_match[2] if len(best_match) > 2 else 1.0)
        counter = Counter(texts)
        best_text = counter.most_common(1)[0][0]
        consensus.append([base_box, best_text, max(scores)])
    return consensus

def tap_option(x, y):
    if not ADB_PATH:
        return False
    x = max(0, min(x, PHONE_W - 1))
    y = max(0, min(y, PHONE_H - 1))
    # 重试3次，第3次重启ADB
    for attempt in range(3):
        try:
            result = subprocess.run(
                [ADB_PATH, "shell", "input", "tap", str(x), str(y)],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except:
            if attempt < 2:
                time.sleep(0.5)
            else:
                # 最后一次尝试，重启ADB
                restart_adb()
                time.sleep(1)
    return False

def wake_screen():
    """唤醒屏幕"""
    if not ADB_PATH:
        return
    try:
        subprocess.run([ADB_PATH, "shell", "input", "keyevent", "224"], capture_output=True, timeout=5)
        subprocess.run([ADB_PATH, "shell", "input", "keyevent", "82"], capture_output=True, timeout=5)
    except:
        pass

def fresh_find_option(ocr_engine, target_label, target_text=None):
    """OCR找到指定选项坐标，重新截图保证实时"""
    frame = capture_frame()
    if frame is None:
        return None
    result = ocr_frame(ocr_engine, frame)
    if not result:
        return None
    for line in result:
        box = line[0]
        text = line[1].strip()
        # 按字母匹配（A. xxx）
        m = re.match(r'^([A-F])\s*[\.、．:：\-]?\s*', text, re.IGNORECASE)
        if m and m.group(1).upper() == target_label.upper():
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            return (cx, cy)
        # 按文本匹配（判断题 正确/错误）
        if target_text and text.strip() == target_text:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            return (cx, cy)
    return None

def fresh_find_button(ocr_engine, keywords):
    """OCR找到包含关键词的按钮坐标，重新截图"""
    frame = capture_frame()
    if frame is None:
        return None
    result = ocr_frame(ocr_engine, frame)
    if not result:
        return None
    for line in result:
        box = line[0]
        text = line[1].strip()
        if any(kw in text for kw in keywords):
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            cx = (min(xs) + max(xs)) // 2
            cy = (min(ys) + max(ys)) // 2
            return (cx, cy)
    return None


class AutoAnswerApp:
    def __init__(self, root):
        global _app_ref
        _app_ref = self
        self.root = root
        self.root.title("答题终结者 T-888型")
        self.root.geometry("860x680")
        self.root.minsize(750, 580)

        self.running = False
        self.log_queue = queue.Queue()
        self.df = None
        self.cache = {}
        self.processed_indices = set()
        self.ocr_engine = None
        self.last_question_key = ""
        self.same_index_counter = {}
        self.first_loop = True
        self.mode = "auto"
        self._repeat_notified = False
        # 坐标系校准缓存：首次OCR后锁定，后续题目复用
        self.calibration = None  # {"roi_x", "roi_y", "ratio_x", "ratio_y"}

        self.create_widgets()
        self.load_config()
        self.process_log_queue()
        # 后台初始化ADB和摄像头，不阻塞界面显示
        threading.Thread(target=_init_adb, daemon=True).start()
        threading.Thread(target=_init_cameras, daemon=True).start()

    def create_widgets(self):
        main = ttk.Frame(self.root, padding="12 10")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 标题
        ttk.Label(main, text="💀 答题终结者 T-888型", font=("微软雅黑", 14, "bold")).pack(anchor="w")
        ttk.Label(main, text="自动模式仅限小米手机，其他品牌请用手动模式", font=("微软雅黑", 8), foreground="gray").pack(anchor="w")

        # === 主体两栏布局 ===
        body = ttk.Frame(main)
        body.pack(fill="both", expand=True, pady=(8, 0))
        body.columnconfigure(0, weight=0, minsize=380)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="n", padx=(0, 10))
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        # ===== 左栏 =====

        # -- 题库管理 --
        f_file = ttk.LabelFrame(left, text=" 题库管理 ", padding="8 6")
        f_file.pack(fill="x", pady=(0, 6))
        f_file.columnconfigure(1, weight=1)

        ttk.Label(f_file, text="题库路径", width=8, anchor="e").grid(row=0, column=0, sticky="e", padx=(0, 4), pady=4)
        self.excel_path = ttk.Entry(f_file)
        self.excel_path.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        bf = ttk.Frame(f_file)
        bf.grid(row=0, column=2)
        ttk.Button(bf, text="浏览", command=self.browse_excel, width=5).pack(side="left", padx=(0, 3))
        ttk.Button(bf, text="加载", command=self.load_excel, width=5).pack(side="left")

        # -- 识别参数 --
        f_ocr = ttk.LabelFrame(left, text=" 识别参数 ", padding="8 6")
        f_ocr.pack(fill="x", pady=(0, 6))
        f_ocr.columnconfigure(1, weight=1)
        f_ocr.columnconfigure(3, weight=1)

        labels_ocr = ["题目匹配", "选项匹配"]
        defaults_ocr = ["0.5", "0"]
        self._param_entries = {}
        for i, (lbl, val) in enumerate(zip(labels_ocr, defaults_ocr)):
            r, c = divmod(i, 2)
            ttk.Label(f_ocr, text=lbl, width=8, anchor="e").grid(row=r, column=c*2, sticky="e", padx=(0, 4), pady=4)
            e = ttk.Entry(f_ocr, width=10, justify="center")
            e.insert(0, val)
            e.grid(row=r, column=c*2+1, sticky="w", pady=4)
            self._param_entries[lbl] = e

        self.ques_thresh_entry = self._param_entries["题目匹配"]
        self.opt_thresh_entry = self._param_entries["选项匹配"]

        # -- 时序参数 --
        f_time = ttk.LabelFrame(left, text=" 时序参数 ", padding="8 6")
        f_time.pack(fill="x", pady=(0, 6))
        f_time.columnconfigure(1, weight=1)
        f_time.columnconfigure(3, weight=1)

        labels_time = ["题前冷却", "题后冷却"]
        defaults_time = ["3.0", "0"]
        for i, (lbl, val) in enumerate(zip(labels_time, defaults_time)):
            r, c = divmod(i, 2)
            ttk.Label(f_time, text=lbl, width=8, anchor="e").grid(row=r, column=c*2, sticky="e", padx=(0, 4), pady=4)
            e = ttk.Entry(f_time, width=10, justify="center")
            e.insert(0, val)
            e.grid(row=r, column=c*2+1, sticky="w", pady=4)
            self._param_entries[lbl] = e

        self.pre_cool_entry = self._param_entries["题前冷却"]
        self.post_cool_entry = self._param_entries["题后冷却"]

        # -- 坐标参数 --
        f_pos = ttk.LabelFrame(left, text=" 坐标参数 ", padding="8 6")
        f_pos.pack(fill="x", pady=(0, 6))
        f_pos.columnconfigure(1, weight=1)
        f_pos.columnconfigure(3, weight=1)

        labels_pos = ["选项偏移", "按钮偏移"]
        defaults_pos = ["200", "200"]
        for i, (lbl, val) in enumerate(zip(labels_pos, defaults_pos)):
            r, c = divmod(i, 2)
            ttk.Label(f_pos, text=lbl, width=8, anchor="e").grid(row=r, column=c*2, sticky="e", padx=(0, 4), pady=4)
            e = ttk.Entry(f_pos, width=10, justify="center")
            e.insert(0, val)
            e.grid(row=r, column=c*2+1, sticky="w", pady=4)
            self._param_entries[lbl] = e

        self.opt_offset_entry = self._param_entries["选项偏移"]
        self.btn_offset_entry = self._param_entries["按钮偏移"]

        # -- 设备参数 --
        f_dev = ttk.LabelFrame(left, text=" 设备参数 ", padding="8 6")
        f_dev.pack(fill="x", pady=(0, 6))
        f_dev.columnconfigure(1, weight=1)

        ttk.Label(f_dev, text="外设选择", width=8, anchor="e").grid(row=0, column=0, sticky="e", padx=(0, 4), pady=4)
        self.cam_var = tk.StringVar()
        self.cam_combo = ttk.Combobox(f_dev, textvariable=self.cam_var, values=["检测中..."], width=12, state="readonly")
        self.cam_combo.grid(row=0, column=1, sticky="w", pady=4)
        self.cam_combo.current(0)
        self.cam_combo.bind("<<ComboboxSelected>>", self.switch_camera)

        ttk.Label(f_dev, text="截图次数", width=8, anchor="e").grid(row=1, column=0, sticky="e", padx=(0, 4), pady=4)
        self.multi_cap_entry = ttk.Entry(f_dev, width=10, justify="center")
        self.multi_cap_entry.insert(0, "1")
        self.multi_cap_entry.grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(f_dev, text="答题模式", width=8, anchor="e").grid(row=2, column=0, sticky="e", padx=(0, 4), pady=4)
        self.mode_var = tk.StringVar(value="auto")
        mode_frame = ttk.Frame(f_dev)
        mode_frame.grid(row=2, column=1, sticky="w", pady=4)
        ttk.Radiobutton(mode_frame, text="自动", variable=self.mode_var, value="auto", command=self._on_mode_change).pack(side="left")
        ttk.Radiobutton(mode_frame, text="手动", variable=self.mode_var, value="manual", command=self._on_mode_change).pack(side="left", padx=(12, 0))

        # -- 操作按钮 --
        f_btn = ttk.LabelFrame(left, text=" 操作 ", padding="8 6")
        f_btn.pack(fill="x")
        f_btn.columnconfigure(0, weight=1)
        f_btn.columnconfigure(1, weight=1)

        self.start_btn = ttk.Button(f_btn, text="开始答题", command=self.start, width=12)
        self.start_btn.grid(row=0, column=0, padx=(0, 3), pady=(0, 3), sticky="ew")
        self.stop_btn = ttk.Button(f_btn, text="停止答题", command=self.stop, width=12, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=(3, 0), pady=(0, 3), sticky="ew")
        ttk.Button(f_btn, text="保存配置", command=self.save_config, width=12).grid(row=1, column=0, padx=(0, 3), pady=(3, 0), sticky="ew")
        ttk.Button(f_btn, text="测试ADB", command=self.test_adb, width=12).grid(row=1, column=1, padx=(3, 0), pady=(3, 0), sticky="ew")

        # ===== 右栏：运行日志 =====
        f_log = ttk.LabelFrame(right, text=" 运行日志 ", padding="8 6")
        f_log.grid(row=0, column=0, sticky="nsew")
        f_log.columnconfigure(0, weight=1)
        f_log.rowconfigure(0, weight=1)

        self.log_area = scrolledtext.ScrolledText(f_log, font=("Consolas", 9), height=20)
        self.log_area.grid(row=0, column=0, sticky="nsew")

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            self.excel_path.insert(0, cfg.get("excel_path", ""))
            self.btn_offset_entry.delete(0, tk.END); self.btn_offset_entry.insert(0, str(cfg.get("btn_offset", 200)))
            self.opt_offset_entry.delete(0, tk.END); self.opt_offset_entry.insert(0, str(cfg.get("opt_offset", 200)))
            self.ques_thresh_entry.delete(0, tk.END); self.ques_thresh_entry.insert(0, str(cfg.get("ques_thresh", 0.5)))
            self.opt_thresh_entry.delete(0, tk.END); self.opt_thresh_entry.insert(0, str(cfg.get("opt_thresh", 0)))
            self.pre_cool_entry.delete(0, tk.END); self.pre_cool_entry.insert(0, str(cfg.get("pre_cool", 3.0)))
            self.post_cool_entry.delete(0, tk.END); self.post_cool_entry.insert(0, str(cfg.get("post_cool", 0)))
            self.multi_cap_entry.delete(0, tk.END); self.multi_cap_entry.insert(0, str(cfg.get("multi_capture", 3)))
            mode = cfg.get("mode", "auto")
            self.mode_var.set(mode)
            self.mode = mode
            cam_idx = cfg.get("camera_index")
            if _cam_inited and AVAILABLE_CAMERAS:
                if cam_idx is not None and cam_idx in AVAILABLE_CAMERAS:
                    self.cam_combo.current(AVAILABLE_CAMERAS.index(cam_idx))
                    self.init_camera(cam_idx)
                else:
                    self.init_camera(AVAILABLE_CAMERAS[0])
            if self.excel_path.get():
                self.load_excel()
        except:
            pass

    def save_config(self):
        try:
            cam_idx = AVAILABLE_CAMERAS[self.cam_combo.current()] if AVAILABLE_CAMERAS else 0
            cfg = {
                "excel_path": self.excel_path.get(),
                "btn_offset": int(self.btn_offset_entry.get().strip() or 200),
                "opt_offset": int(self.opt_offset_entry.get().strip() or 200),
                "ques_thresh": float(self.ques_thresh_entry.get().strip() or 0.5),
                "opt_thresh": float(self.opt_thresh_entry.get().strip() or 0),
                "pre_cool": float(self.pre_cool_entry.get().strip() or 3.0),
                "post_cool": float(self.post_cool_entry.get().strip() or 0),
                "multi_capture": int(self.multi_cap_entry.get().strip() or 3),
                "mode": self.mode_var.get(),
                "camera_index": cam_idx
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            self.log("保存失败")

    def browse_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if path:
            self.excel_path.delete(0, tk.END)
            self.excel_path.insert(0, path)
            self.load_excel()

    def switch_camera(self, event=None):
        global cap
        if not AVAILABLE_CAMERAS:
            return
        idx = self.cam_combo.current()
        cam_idx = AVAILABLE_CAMERAS[idx]
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(cam_idx)

    def init_camera(self, cam_idx):
        global cap
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(cam_idx)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return True
        return False

    def test_adb(self):
        if not ADB_PATH:
            self.log("未找到adb.exe")
            return
        try:
            # 先尝试重启ADB服务
            self.log("正在测试ADB连接...")
            subprocess.run([ADB_PATH, "kill-server"], capture_output=True, timeout=5)
            subprocess.run([ADB_PATH, "start-server"], capture_output=True, timeout=10)
            
            res = subprocess.run([ADB_PATH, "devices"], capture_output=True, text=True, timeout=10)
            lines = [l for l in res.stdout.strip().split('\n')[1:] if l.strip()]
            if not lines:
                self.log("未检测到设备")
                return
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    state = parts[1]
                    if state == "unauthorized":
                        self.log("设备未授权")
                        return
                    elif state == "offline":
                        self.log("设备离线")
                        return
            test = subprocess.run([ADB_PATH, "shell", "getprop", "ro.product.model"], capture_output=True, text=True, timeout=10)
            if test.returncode == 0 and test.stdout.strip():
                self.log(f"ADB正常: {test.stdout.strip()}")
            else:
                self.log("ADB异常")
        except subprocess.TimeoutExpired:
            self.log("ADB超时")
            try:
                subprocess.run([ADB_PATH, "kill-server"], capture_output=True, timeout=5)
                subprocess.run([ADB_PATH, "start-server"], capture_output=True, timeout=10)
                self.log("ADB服务已重启")
            except:
                self.log("无法重启ADB服务")
        except Exception as e:
            self.log("ADB异常")

    def log(self, msg):
        self.log_queue.put(msg)

    def process_log_queue(self):
        while not self.log_queue.empty():
            self.log_area.insert(tk.END, self.log_queue.get() + "\n")
            self.log_area.see(tk.END)
        self.root.after(100, self.process_log_queue)

    def load_excel(self):
        try:
            self.log("正在加载题库...")
            df = pd.read_excel(self.excel_path.get())
            if df.shape[0] == 0:
                messagebox.showerror("错误", "Excel 为空")
                return
            self.df = df.iloc[:, :3].copy()
            self.df.columns = ['question', 'options_str', 'answer_letter']
            self.cache = {}
            self.processed_indices.clear()
            self.last_question_key = ""
            self.same_index_counter.clear()
            self.first_loop = True
            self.log(f"题库加载完成")
        except Exception as e:
            messagebox.showerror("错误", f"读取失败: {e}")

    def parse_options_with_letters(self, opt_str):
        if not opt_str or str(opt_str).strip() in ['nan', '']:
            return {}
        s = str(opt_str).strip()
        result = {}
        if '|' in s:
            parts = s.split('|')
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                m = re.match(r'^([A-F])\s*[-\.、．:：]?\s*(.*)', part, re.IGNORECASE)
                if m:
                    result[m.group(1).upper()] = m.group(2).strip()
                else:
                    m2 = re.match(r'^([A-F])\s*(.*)', part, re.IGNORECASE)
                    if m2:
                        result[m2.group(1).upper()] = m2.group(2).strip()
            if result:
                return result
        pattern = re.compile(r'([A-F])\s*[\.、．:：\-]\s*([^A-F]*?)(?=\s*[A-F]\s*[\.、．:：\-]|$)', re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(s)
        if matches:
            for letter, content in matches:
                letter = letter.strip().upper()
                content = content.strip()
                if content:
                    result[letter] = content
            return result
        return result

    def get_correct_texts(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        row = self.df.iloc[idx]
        ans_str = str(row['answer_letter']).strip()
        opt_map = self.parse_options_with_letters(str(row['options_str']))
        texts = []

        # 判断题：选项只有2个且内容为正确/错误/对/错
        is_judge = False
        if len(opt_map) == 2:
            vals = set(opt_map.values())
            if vals <= {'正确', '错误', '对', '错', '√', '×'}:
                is_judge = True

        if is_judge:
            if ans_str.upper() == 'A':
                texts = [opt_map.get('A', '正确')]
            elif ans_str.upper() == 'B':
                texts = [opt_map.get('B', '错误')]
        elif re.search(r'[A-F]', ans_str, re.IGNORECASE):
            raw = ans_str.upper().replace('.', '').replace(' ', '')
            letters = re.split(r'[,，\s]+', raw)
            if len(letters) == 1 and len(letters[0]) > 1:
                letters = list(letters[0])
            for l in letters:
                if l in opt_map:
                    texts.append(opt_map[l])
                else:
                    texts.append(l)
        else:
            texts = [ans_str] if ans_str else []

        if not texts and ans_str:
            texts = [ans_str]
        self.cache[idx] = texts
        return texts

    def clean_text(self, text):
        # 去OCR噪声：多余空格、全角/半角混用的标点、引号
        # 保留数字、中文、字母用于匹配
        text = re.sub(r'[\s\.\,\。\，\：\、\（\）\《\》\「\」\-\_\?\？\!\！\~\～\'\"\']', '', text)
        return text

    def strip_option_prefix(self, text):
        """去掉选项字母前缀 (如 'A. xxx' → 'xxx')，用于与题库选项内容对齐"""
        m = re.match(r'^[A-F]\s*[\.、．:：\-]?\s*(.*)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text

    def match_question(self, ocr_q, threshold):
        best_score = 0
        best_idx = None
        clean_ocr = self.clean_text(ocr_q)
        if not clean_ocr or len(clean_ocr) < 3:
            return None, 0

        for idx, row in self.df.iterrows():
            clean_db = self.clean_text(str(row['question']))
            if not clean_db or len(clean_db) < 3:
                continue

            score1 = SequenceMatcher(None, clean_ocr, clean_db).ratio()

            # 包含关系加分
            score2 = 0
            if clean_ocr in clean_db:
                score2 = 0.8 * (len(clean_ocr) / len(clean_db))
            elif clean_db in clean_ocr:
                score2 = 0.8 * (len(clean_db) / len(clean_ocr))

            # 前缀匹配
            prefix_len = min(15, len(clean_ocr), len(clean_db))
            score3 = 1.0 if clean_ocr[:prefix_len] == clean_db[:prefix_len] else 0

            score = max(score1, score2, score3 * 0.9)

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx, best_score

    def start(self):
        if self.df is None:
            messagebox.showerror("错误", "请先加载题库")
            return
        # 确保摄像头已打开
        if cap is None or not cap.isOpened():
            # 等待摄像头检测完成（最多2秒）
            wait = 0
            while not _cam_inited and wait < 20:
                time.sleep(0.1)
                wait += 1
            if AVAILABLE_CAMERAS:
                self.init_camera(AVAILABLE_CAMERAS[0])
            if cap is None or not cap.isOpened():
                messagebox.showerror("错误", "摄像头不可用，请检查连接")
                return
        # 确保ADB已初始化
        wait = 0
        while not _adb_inited and wait < 20:
            time.sleep(0.1)
            wait += 1
        self.mode = self.mode_var.get()
        self.running = True
        self.processed_indices.clear()
        self.last_question_key = ""
        self.same_index_counter.clear()
        self.first_loop = True
        self._repeat_notified = False
        self.cache.clear()
        self.calibration = None  # 重置坐标校准
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.run_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _on_mode_change(self):
        self.mode = self.mode_var.get()

    def _refresh_cameras(self):
        """后台摄像头检测完成后刷新下拉框"""
        if AVAILABLE_CAMERAS:
            labels = [f"摄像头 {i}" for i in AVAILABLE_CAMERAS]
            self.cam_combo.config(values=labels)
            self.cam_combo.current(0)

    def run_loop(self):
        try:
            # 延迟初始化OCR引擎，减少启动时间
            if self.ocr_engine is None:
                self.log("正在加载OCR引擎...")
                self.ocr_engine = RapidOCR()
                self.log("OCR加载完成")

            while self.running:
                try:
                    try:
                        ques_thresh = float(self.ques_thresh_entry.get().strip() or 0.5)
                    except:
                        ques_thresh = 0.5
                    try:
                        opt_thresh = float(self.opt_thresh_entry.get().strip() or 0)
                    except:
                        opt_thresh = 0
                    try:
                        pre_cool = float(self.pre_cool_entry.get().strip() or 3.0)
                    except:
                        pre_cool = 3.0
                    try:
                        post_cool = float(self.post_cool_entry.get().strip() or 0)
                    except:
                        post_cool = 0
                    pre_cool = max(0.5, pre_cool)
                    post_cool = max(0, post_cool)

                    try:
                        multi_cap = int(self.multi_cap_entry.get().strip() or 3)
                    except:
                        multi_cap = 3
                    multi_cap = max(1, min(multi_cap, 5))

                    opt_offset = int(self.opt_offset_entry.get().strip() or 200)
                    btn_offset = int(self.btn_offset_entry.get().strip() or 200)

                    # === 搜题重试循环：截图OCR+搜题库，最多3次 ===
                    wake_screen()  # 每道题唤醒一次
                    if self.first_loop:
                        time.sleep(1)  # 首题多等1秒，等摄像头就绪
                    best_idx = None
                    score = 0
                    question = ""
                    current_key = ""
                    options = []
                    q_blocks = []
                    next_btn = None
                    items = []
                    roi_x = roi_y = ratio_x = ratio_y = 0
                    first_opt_top = None

                    for search_attempt in range(3):
                        cur_frame = capture_frame()
                        if cur_frame is None:
                            time.sleep(0.1)
                            continue
                        result = ocr_frame(self.ocr_engine, cur_frame)
                        if not result:
                            time.sleep(0.1)
                            continue

                        items = []
                        for line in result:
                            box = line[0]
                            text = line[1]
                            score_val = line[2] if len(line) > 2 else 1.0
                            xs = [pt[0] for pt in box]
                            ys = [pt[1] for pt in box]
                            items.append({
                                "text": text,
                                "x": min(xs), "y": min(ys),
                                "w": max(xs)-min(xs), "h": max(ys)-min(ys),
                                "score": score_val
                            })

                        # 合并相邻的选项字母和内容
                        merged = []
                        skip = set()
                        for i, item in enumerate(items):
                            if i in skip:
                                continue
                            if re.match(r'^[A-F]$', item['text'], re.IGNORECASE) and i+1 < len(items):
                                nxt = items[i+1]
                                if nxt['y'] - item['y'] < 30:
                                    merged.append({
                                        "text": item['text'] + ". " + nxt['text'],
                                        "x": item['x'], "y": item['y'],
                                        "w": nxt['x']+nxt['w']-item['x'],
                                        "h": nxt['y']+nxt['h']-item['y'],
                                        "score": min(item.get('score', 1), nxt.get('score', 1))
                                    })
                                    skip.add(i+1)
                                    continue
                            merged.append(item)
                        items = merged

                        options = []
                        q_blocks = []
                        next_btn = None
                        judge_keywords = {'对', '错', '正确', '错误', '√', '×'}

                        for item in items:
                            txt = item['text'].strip()
                            if '下一题' in txt or '下一页' in txt or '下一' in txt:
                                next_btn = (item['x'] + item['w']//2, item['y'] + item['h']//2)
                                continue
                            m = re.match(r'^([A-F])\s*[\.、．:：\-]?\s*(.*)', txt, re.IGNORECASE)
                            if m:
                                options.append((m.group(1).upper(), m.group(2), (item['x']+item['w']//2, item['y']+item['h']//2)))
                            elif txt in judge_keywords:
                                options.append(('J', txt, (item['x']+item['w']//2, item['y']+item['h']//2)))
                            else:
                                q_blocks.append(item)

                        if not options:
                            time.sleep(0.1)
                            continue

                        img = cur_frame
                        if img is None:
                            time.sleep(0.1)
                            continue
                        h, w = img.shape[:2]
                        all_y = [it['y'] for it in items]
                        all_x = [it['x'] for it in items]
                        min_y = min(all_y); max_y = max([it['y']+it['h'] for it in items])
                        min_x = min(all_x); max_x = max([it['x']+it['w'] for it in items])

                        if self.calibration is None:
                            # 首次：建立校准映射
                            roi_y = max(0, min_y - 200)
                            roi_x = max(0, min_x - 150)
                            roi_h = min(h - roi_y, max_y - min_y + 400)
                            roi_w = min(w - roi_x, max_x - min_x + 300)
                            ratio_x = PHONE_W / roi_w if roi_w else 1
                            ratio_y = PHONE_H / roi_h if roi_h else 1
                            self.calibration = {
                                "roi_x": roi_x, "roi_y": roi_y,
                                "ratio_x": ratio_x, "ratio_y": ratio_y
                            }
                        else:
                            # 后续：复用校准，不做漂移
                            roi_x = self.calibration["roi_x"]
                            roi_y = self.calibration["roi_y"]
                            ratio_x = self.calibration["ratio_x"]
                            ratio_y = self.calibration["ratio_y"]

                        first_opt_top = None
                        for item in items:
                            if re.match(r'^[A-F]', item['text'], re.IGNORECASE) or item['text'] in judge_keywords:
                                first_opt_top = item['y']
                                break
                        if first_opt_top is None:
                            first_opt_top = min_y
                        question = " ".join([it['text'] for it in q_blocks if it['y'] < first_opt_top])
                        if not question:
                            question = " ".join([it['text'] for it in q_blocks[:3]])

                        if not question:
                            time.sleep(0.1)
                            continue

                        current_key = self.clean_text(question)[:50]

                        if current_key == self.last_question_key:
                            break

                        # 搜题库
                        best_idx, score = self.match_question(question, ques_thresh)

                        if best_idx is not None and score >= ques_thresh:
                            break

                        # 降级重试
                        retry_idx, retry_score = self.match_question(question, max(0.2, ques_thresh * 0.6))
                        if retry_idx is not None and retry_score > 0.2:
                            best_idx = retry_idx
                            score = retry_score
                            break

                        time.sleep(0.1)

                    # 3次都没搜到 → 暂停转手动
                    if best_idx is None:
                        _to_manual(self)
                        time.sleep(1)
                        continue

                    self.last_question_key = current_key

                    if best_idx in self.processed_indices:
                        self.same_index_counter[best_idx] = self.same_index_counter.get(best_idx, 0) + 1
                        if self.same_index_counter[best_idx] >= 3:
                            self.same_index_counter[best_idx] = 0
                        if self.mode == "auto":
                            if next_btn:
                                nx, ny = next_btn
                                pb_x = int((nx - roi_x) * ratio_x)
                                pb_y = int((ny - roi_y) * ratio_y) + btn_offset
                                pb_x = max(0, min(pb_x, PHONE_W - 1))
                                pb_y = max(0, min(pb_y, PHONE_H - 1))

                                tap_option(pb_x, pb_y)
                            time.sleep(pre_cool)
                        else:
                            if not self._repeat_notified:
                                self._repeat_notified = True
                            time.sleep(1)
                        self.last_question_key = ""
                        self.same_index_counter.clear()
                        continue
                    else:
                        self._repeat_notified = False

                    self.processed_indices.add(best_idx)
                    self.same_index_counter[best_idx] = 0

                    correct_texts = self.get_correct_texts(best_idx)
                    if not correct_texts:
                        row = self.df.iloc[best_idx]
                        raw_ans = str(row['answer_letter']).strip()
                        if raw_ans:
                            correct_texts = [raw_ans]
                        else:
                            _to_manual(self)
                            time.sleep(1)
                            continue

                    all_opts = []
                    for label, text, (x, y) in options:
                        clean = self.clean_text(text)
                        if not clean:
                            continue
                        all_opts.append((label, clean, x-30, y))

                    matched_opts = []

                    # 从题库获取正确答案
                    targets = [self.clean_text(t) for t in correct_texts if t]
                    if not targets:
                        _to_manual(self)
                        time.sleep(1)
                        continue

                    opt_list = []
                    seen_labels = set()
                    for label, clean, x, y in all_opts:
                        if label not in seen_labels:
                            opt_list.append((label, clean, x, y))
                            seen_labels.add(label)

                    if not opt_list:
                        _to_manual(self)
                        time.sleep(1)
                        continue

                    selected_labels = set()

                    for target in targets:
                        best_opt = None
                        best_sim = -1
                        target_stripped = self.strip_option_prefix(target) if re.match(r'^[A-F]', target, re.IGNORECASE) else target

                        for label, clean, x, y in opt_list:
                            if label in selected_labels:
                                continue
                            sim = 0
                            if clean == target or clean == target_stripped:
                                sim = 1.0
                            elif target in clean or clean in target or target_stripped in clean or clean in target_stripped:
                                sim = 0.9
                            else:
                                sim = SequenceMatcher(None, target_stripped, clean).ratio()
                                keywords = re.split(r'[\s,，、.。;；:：]', target_stripped)
                                keywords = [k for k in keywords if len(k) > 1]
                                if keywords:
                                    longest = max(keywords, key=len)
                                    if longest in clean:
                                        sim = max(sim, 0.75)
                                max_len = max(len(target_stripped), len(clean))
                                min_len = min(len(target_stripped), len(clean))
                                if max_len > 0 and min_len / max_len > 0.85:
                                    sim = max(sim, 0.85)

                            if sim > best_sim:
                                best_sim = sim
                                best_opt = (label, x, y)

                        if best_opt and best_sim >= opt_thresh:
                            matched_opts.append(best_opt)
                            selected_labels.add(best_opt[0])
                        else:
                            _to_manual(self)
                            time.sleep(1)
                            break

                    if len(targets) > 1:
                        matched_count = len([m for m in matched_opts if m[0] != 'J'])
                        if matched_count < len(targets):
                            self.log(f"多选匹配不完整({matched_count}/{len(targets)})，转手动")
                            _to_manual(self)
                            time.sleep(1)
                            continue

                    # 答案显示：字母用试卷OCR识别的，内容用题库的
                    # matched_opts[i] 与 correct_texts[i] 按顺序对应
                    if matched_opts:
                        display_parts = []
                        for i, (label, x, y) in enumerate(matched_opts):
                            opt_text = correct_texts[i].strip() if i < len(correct_texts) else ""
                            display_parts.append(f"{label}. {opt_text}" if opt_text else label)
                        answer_str = ", ".join(display_parts)
                        self.log(f"答案: {answer_str}")
                    else:
                        self.log("未匹配到选项，转手动")
                        _to_manual(self)
                        time.sleep(1)
                        continue

                    if self.mode == "auto":
                        if not ADB_PATH:
                            self.log("未找到ADB")
                            self.mode = "manual"
                            self.mode_var.set("manual")
                            time.sleep(1)
                            continue

                        # 每个选项点击前重新截图定位
                        for label, x, y in matched_opts:
                            # 重新截图找此选项的最新坐标
                            # 判断题用文本匹配兜底
                            target_text = None
                            if label == 'J':
                                row = self.df.iloc[best_idx]
                                opt_map = self.parse_options_with_letters(str(row['options_str']))
                                ans_str = str(row['answer_letter']).strip().upper()
                                target_text = opt_map.get(ans_str, "")
                            fresh_pos = fresh_find_option(self.ocr_engine, label, target_text)
                            if fresh_pos:
                                cx, cy = fresh_pos
                                ph_x = int((cx - roi_x) * ratio_x)
                                ph_y = int((cy - roi_y) * ratio_y) + opt_offset
                                ph_x = max(0, min(ph_x, PHONE_W - 1))
                                ph_y = max(0, min(ph_y, PHONE_H - 1))

                                tap_option(ph_x, ph_y)
                            else:
                                # 找不到就用旧坐标兜底
                                ph_x = int((x - roi_x) * ratio_x)
                                ph_y = int((y - roi_y) * ratio_y) + opt_offset
                                ph_x = max(0, min(ph_x, PHONE_W - 1))
                                ph_y = max(0, min(ph_y, PHONE_H - 1))

                                tap_option(ph_x, ph_y)
                            time.sleep(0.1)

                        time.sleep(0.1)

                        # 重新截图找下一题按钮
                        fresh_next = fresh_find_button(self.ocr_engine, ['下一题', '下一页', '下一'])
                        if fresh_next:
                            cx, cy = fresh_next
                            pb_x = int((cx - roi_x) * ratio_x)
                            pb_y = int((cy - roi_y) * ratio_y) + btn_offset
                            pb_x = max(0, min(pb_x, PHONE_W - 1))
                            pb_y = max(0, min(pb_y, PHONE_H - 1))
                            tap_option(pb_x, pb_y)
                        elif next_btn:
                            # 用旧坐标兜底
                            nx, ny = next_btn
                            pb_x = int((nx - roi_x) * ratio_x)
                            pb_y = int((ny - roi_y) * ratio_y) + btn_offset
                            pb_x = max(0, min(pb_x, PHONE_W - 1))
                            pb_y = max(0, min(pb_y, PHONE_H - 1))
                            tap_option(pb_x, pb_y)

                        time.sleep(pre_cool)
                    else:
                        time.sleep(1)

                    self.last_question_key = ""
                    self.same_index_counter.clear()
                    self._repeat_notified = False
                    self.first_loop = False

                except Exception as e:
                    _to_manual(self)
                    wake_screen()
                    time.sleep(1)

        except Exception as e:
            self.log("异常")
            wake_screen()

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoAnswerApp(root)
    root.mainloop()
