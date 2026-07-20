import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
import threading
import queue
import json
import os
import sys
import time
import re
import gc
import subprocess
from difflib import SequenceMatcher
import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from collections import Counter

CONFIG_FILE = "answer_config.json"
_CLEAN_RE = re.compile(r'[\s\.\,\。\，\：\、\（\）\《\》\「\」\-\_\?\？\!\！\~\～\'\"\']')
_JUDGE_KW = {'对', '错', '正确', '错误', '√', '×', '○', '●', '◯', '•', '·'}

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

def _to_manual(app, reason=""):
    msg = "手动模式"
    if reason:
        msg += f"：{reason}"
    app.log(msg)
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
    for attempt in range(3):
        try:
            subprocess.Popen(
                [ADB_PATH, "shell", "input", "tap", str(x), str(y)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True
        except:
            if attempt >= 2:
                restart_adb()
            time.sleep(0.3)
    return False

def wake_screen():
    """保持屏幕常亮"""
    if not ADB_PATH:
        return
    try:
        subprocess.run([ADB_PATH, "shell", "svc", "power", "stayon", "usb"], capture_output=True, timeout=3)
        subprocess.run([ADB_PATH, "shell", "settings", "put", "system", "screen_off_timeout", "2147483647"], capture_output=True, timeout=3)
    except:
        pass

class AutoAnswerApp:
    def __init__(self, root):
        global _app_ref
        _app_ref = self
        self.root = root
        self.root.title("答题终结者")
        self.root.geometry("860x680")
        self.root.minsize(750, 580)

        self.running = False
        self.log_queue = queue.Queue()
        self.df = None
        self.cache = {}
        self.processed_indices = set()
        self.ocr_engine = None
        self.last_question_key = ""
        self.first_loop = True
        self.mode = "auto"
        self.calibration = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
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

        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 10))
        right = ttk.Frame(body)
        right.pack(fill="both", expand=True)

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

        ttk.Label(f_dev, text="答题模式", width=8, anchor="e").grid(row=1, column=0, sticky="e", padx=(0, 4), pady=4)
        self.mode_var = tk.StringVar(value="auto")
        mode_frame = ttk.Frame(f_dev)
        mode_frame.grid(row=1, column=1, sticky="w", pady=4)
        ttk.Radiobutton(mode_frame, text="自动", variable=self.mode_var, value="auto").pack(side="left")
        ttk.Radiobutton(mode_frame, text="手动", variable=self.mode_var, value="manual").pack(side="left", padx=(12, 0))

        self.consensus_var = tk.BooleanVar(value=False)
        ttk.Label(f_dev, text="多次识别", width=8, anchor="e").grid(row=2, column=0, sticky="e", padx=(0, 4), pady=4)
        ttk.Checkbutton(f_dev, text="截3次取众数,更准但慢", variable=self.consensus_var).grid(row=2, column=1, sticky="w", pady=4)

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
        f_log.pack(fill="both", expand=True)

        self.log_area = scrolledtext.ScrolledText(f_log, font=("Consolas", 9))
        self.log_area.pack(fill="both", expand=True)

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
            self.first_loop = True
            self._clean_questions = [self.clean_text(str(q)) for q in self.df['question']]
            self.log("题库加载完成")
        except Exception as e:
            messagebox.showerror("错误", f"读取失败: {e}")

    def parse_options_with_letters(self, opt_str):
        if not opt_str or str(opt_str).strip() in ['nan', '']:
            return {}
        s = str(opt_str).strip()
        result = {}
        # 按常见分隔符拆分
        for sep in ['|', '\n', '；', ';']:
            if sep in s:
                parts = s.split(sep)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    m = re.match(r'^([A-F])\s*[-\.、．:：（\(]?\s*(.*)', part, re.IGNORECASE)
                    if m:
                        result[m.group(1).upper()] = m.group(2).strip().rstrip('）\)')
                if result:
                    return result
        # 正则提取 A.xxx B.xxx 格式
        pattern = re.compile(r'([A-F])\s*[\.、．:：\-（\(]\s*([^A-F]*?)(?=\s*[A-F]\s*[\.、．:：\-（\(]|$)', re.IGNORECASE | re.DOTALL)
        matches = pattern.findall(s)
        if matches:
            for letter, content in matches:
                letter = letter.strip().upper()
                content = content.strip().rstrip('）\)')
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
                if l in opt_map and opt_map[l]:
                    texts.append(opt_map[l])
        else:
            texts = [ans_str] if ans_str else []
        self.cache[idx] = texts
        return texts

    def clean_text(self, text):
        return _CLEAN_RE.sub('', text)

    def match_question(self, ocr_q, threshold):
        best_score = 0
        best_idx = None
        clean_ocr = self.clean_text(ocr_q)
        if not clean_ocr or len(clean_ocr) < 3:
            return None, 0

        len_ocr = len(clean_ocr)
        for idx, clean_db in enumerate(self._clean_questions):
            if len(clean_db) < 3:
                continue

            score1 = SequenceMatcher(None, clean_ocr, clean_db).ratio()

            score2 = 0
            if clean_ocr in clean_db:
                score2 = 0.7 + 0.3 * (len_ocr / len(clean_db))
            elif clean_db in clean_ocr:
                score2 = 0.7 + 0.3 * (len(clean_db) / len_ocr)

            match_len = 0
            for a, b in zip(clean_ocr, clean_db):
                if a == b:
                    match_len += 1
                else:
                    break
            min_len = min(len_ocr, len(clean_db))
            score3 = match_len / min_len if min_len > 0 else 0

            score = max(score1, score2, score3)

            if score > best_score:
                best_score = score
                best_idx = idx
                if score >= 1.0:
                    break

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
        self.running = True
        self.processed_indices.clear()
        self.last_question_key = ""
        self.first_loop = True
        self.cache.clear()
        self.calibration = None
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.run_loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _on_closing(self):
        self.running = False
        global cap
        if cap is not None:
            cap.release()
            cap = None
        if self.ocr_engine is not None:
            try:
                del self.ocr_engine
            except:
                pass
            self.ocr_engine = None
            gc.collect()
        self.root.destroy()
        self.root.after(200, lambda: os._exit(0))

    def _refresh_cameras(self):
        """后台摄像头检测完成后刷新下拉框"""
        if AVAILABLE_CAMERAS:
            labels = [f"摄像头 {i}" for i in AVAILABLE_CAMERAS]
            self.cam_combo.config(values=labels)
            self.cam_combo.current(0)

    def run_loop(self):
        try:
            if self.ocr_engine is None:
                self.log("正在加载OCR引擎...")
                self.ocr_engine = RapidOCR()
                self.log("OCR加载完成")

            while self.running:
                try:
                    # 实时读取UI状态
                    self.mode = self.mode_var.get()
                    ques_thresh = float(self.ques_thresh_entry.get().strip() or 0.5)
                    opt_thresh = float(self.opt_thresh_entry.get().strip() or 0)
                    pre_cool = max(0.5, float(self.pre_cool_entry.get().strip() or 3.0))
                    opt_offset = int(self.opt_offset_entry.get().strip() or 200)
                    btn_offset = int(self.btn_offset_entry.get().strip() or 200)

                    # 每题唤醒屏幕
                    wake_screen()
                    if self.first_loop:
                        time.sleep(3)

                    # === 截图OCR ===
                    best_idx = None
                    question = ""
                    current_key = ""
                    options = []
                    q_blocks = []
                    next_btn = None
                    items = []
                    roi_x = roi_y = ratio_x = ratio_y = 0

                    for search_attempt in range(3):
                        # 始终截一张用于校准
                        cur_frame = capture_frame()
                        if cur_frame is None:
                            time.sleep(0.1)
                            continue

                        if self.consensus_var.get():
                            result = ocr_with_consensus(self.ocr_engine, times=3)
                        else:
                            result = ocr_frame(self.ocr_engine, cur_frame)
                        if not result:
                            time.sleep(0.1)
                            continue

                        # 解析OCR结果
                        items = []
                        for line in result:
                            box = line[0]
                            text = line[1]
                            xs = [pt[0] for pt in box]
                            ys = [pt[1] for pt in box]
                            items.append({"text": text, "x": min(xs), "y": min(ys),
                                          "w": max(xs)-min(xs), "h": max(ys)-min(ys)})

                        # 合并相邻项
                        merged = []
                        skip = set()
                        for i, item in enumerate(items):
                            if i in skip:
                                continue
                            txt = item['text']
                            # A/B/C字母 + 下方内容合并
                            if re.match(r'^[A-F]$', txt, re.IGNORECASE) and i+1 < len(items):
                                nxt = items[i+1]
                                if nxt['y'] - item['y'] < 30:
                                    merged.append({"text": txt + ". " + nxt['text'],
                                                   "x": item['x'], "y": item['y'],
                                                   "w": nxt['x']+nxt['w']-item['x'],
                                                   "h": nxt['y']+nxt['h']-item['y']})
                                    skip.add(i+1)
                                    continue
                            # 圆圈 + 判断题文字合并
                            if txt in {'○', '●', '◯', '•', '·'} and i+1 < len(items):
                                nxt = items[i+1]
                                if nxt['y'] - item['y'] < 30 and nxt['text'] not in {'○', '●', '◯', '•', '·'}:
                                    merged.append({"text": nxt['text'],
                                                   "x": item['x'], "y": item['y'],
                                                   "w": nxt['x']+nxt['w']-item['x'],
                                                   "h": nxt['y']+nxt['h']-item['y']})
                                    skip.add(i+1)
                                    continue
                            merged.append(item)
                        items = merged

                        # 分类：选项 / 题目 / 下一题按钮
                        options = []
                        q_blocks = []
                        next_btn = None
                        for item in items:
                            txt = item['text'].strip()
                            if '下一题' in txt or '下一页' in txt or '下一' in txt:
                                next_btn = (item['x'] + item['w']//2, item['y'] + item['h']//2)
                                continue
                            m = re.match(r'^([A-F])\s*[\.、．:：\-]?\s*(.*)', txt, re.IGNORECASE)
                            if m:
                                cx, cy = item['x'] + item['w']//2, item['y'] + item['h']//2
                                options.append((m.group(1).upper(), m.group(2), (cx, cy)))
                            elif txt in _JUDGE_KW:
                                cx, cy = item['x'] + item['w']//2, item['y'] + item['h']//2
                                options.append(('J', txt, (cx, cy)))
                            else:
                                q_blocks.append(item)

                        if not options:
                            time.sleep(0.1)
                            continue

                        # 坐标校准
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
                            roi_y = max(0, min_y - 200)
                            roi_x = max(0, min_x - 150)
                            roi_h = min(h - roi_y, max_y - min_y + 400)
                            roi_w = min(w - roi_x, max_x - min_x + 300)
                            self.calibration = {"roi_x": roi_x, "roi_y": roi_y,
                                                 "ratio_x": PHONE_W / max(roi_w, 1),
                                                 "ratio_y": PHONE_H / max(roi_h, 1)}
                        roi_x = self.calibration["roi_x"]
                        roi_y = self.calibration["roi_y"]
                        ratio_x = self.calibration["ratio_x"]
                        ratio_y = self.calibration["ratio_y"]

                        # 提取题目文本
                        first_opt_top = None
                        for item in items:
                            if re.match(r'^[A-F]', item['text'], re.IGNORECASE) or item['text'] in _JUDGE_KW:
                                first_opt_top = item['y']
                                break
                        first_opt_top = first_opt_top or min_y
                        question = " ".join([it['text'] for it in q_blocks if it['y'] < first_opt_top])
                        if not question:
                            question = " ".join([it['text'] for it in q_blocks[:3]])
                        if not question:
                            time.sleep(0.1)
                            continue

                        current_key = self.clean_text(question)[:50]
                        if current_key == self.last_question_key:
                            break

                        best_idx, _ = self.match_question(question, ques_thresh)
                        if best_idx is not None:
                            break

                        # 降级搜题
                        retry_idx, retry_score = self.match_question(question, max(0.2, ques_thresh * 0.6))
                        if retry_idx is not None and retry_score > 0.2:
                            best_idx = retry_idx
                            break

                        time.sleep(0.1)

                    if best_idx is None:
                        _to_manual(self, "3次搜题未匹配")
                        time.sleep(1)
                        continue

                    self.last_question_key = current_key

                    # === 选项匹配 ===
                    correct_texts = self.get_correct_texts(best_idx)
                    if not correct_texts:
                        _to_manual(self, "题库无答案")
                        time.sleep(1)
                        continue

                    targets = [self.clean_text(t) for t in correct_texts if t]
                    if not targets:
                        _to_manual(self, "答案内容为空")
                        time.sleep(1)
                        continue

                    # 去重选项列表
                    opt_list = []
                    seen_labels = set()
                    for label, text, (x, y) in options:
                        clean = self.clean_text(text)
                        if not clean:
                            continue
                        if label not in seen_labels or label == 'J':
                            opt_list.append((label, clean, x, y))
                            seen_labels.add(label)

                    if not opt_list:
                        _to_manual(self, "屏幕无选项")
                        time.sleep(1)
                        continue

                    # 逐个目标匹配最优选项
                    matched_opts = []
                    used_pos = set()
                    match_ok = True
                    for target in targets:
                        best_opt = None
                        best_sim = -1
                        for label, clean, x, y in opt_list:
                            if (x, y) in used_pos:
                                continue
                            sim = 0
                            if clean == target:
                                sim = 1.0
                            elif len(target) >= 2 and len(clean) >= 2:
                                if target in clean:
                                    sim = 0.95
                                elif clean in target:
                                    sim = 0.9
                            if sim < 0.9:
                                seq = SequenceMatcher(None, target, clean).ratio()
                                sim = max(sim, seq)
                                if target and clean and target[0] == clean[0]:
                                    sim = max(sim, min(seq + 0.1, 0.95))
                                ml = max(len(target), len(clean))
                                nl = min(len(target), len(clean))
                                if ml > 0 and nl / ml > 0.8:
                                    sim = max(sim, min(seq + 0.15, 0.95))
                            if sim > best_sim:
                                best_sim = sim
                                best_opt = (label, x, y)
                        # 多选要求至少0.6置信度，单选至少0.5
                        min_conf = 0.6 if len(targets) > 1 else 0.5
                        if best_opt and best_sim >= max(min_conf, opt_thresh):
                            matched_opts.append(best_opt)
                            used_pos.add((best_opt[1], best_opt[2]))
                        else:
                            match_ok = False
                            break

                    # 多选必须全部匹配成功
                    if len(targets) > 1 and len(matched_opts) != len(targets):
                        match_ok = False

                    if not match_ok or not matched_opts:
                        _to_manual(self, "选项匹配度不足")
                        time.sleep(1)
                        continue

                    # === 重复识别：跳过 ===
                    if best_idx in self.processed_indices:
                        if self.mode == "auto":
                            if len(correct_texts) <= 1:
                                for _, x, y in matched_opts:
                                    tap_option(int((x - roi_x) * ratio_x),
                                               int((y - roi_y) * ratio_y) + opt_offset)
                                    time.sleep(0.5)
                            if next_btn:
                                nx, ny = next_btn
                                tap_option(int((nx - roi_x) * ratio_x),
                                           int((ny - roi_y) * ratio_y) + btn_offset)
                            time.sleep(pre_cool)
                        else:
                            time.sleep(1)
                        self.last_question_key = ""
                        continue

                    # === 新题 ===
                    self.processed_indices.add(best_idx)
                    is_first = self.first_loop
                    if is_first:
                        self.log("首题已识别，手动翻页后开始自动答题")
                        self.first_loop = False

                    # 答案显示
                    used_labels = set()
                    parts = []
                    for ct in correct_texts:
                        ct_clean = self.clean_text(ct)
                        found = None
                        for lb, tx, *_ in options:
                            if not tx or (lb in used_labels and lb != 'J'):
                                continue
                            if self.clean_text(tx) == ct_clean:
                                found = lb
                                break
                        if not found and len(ct_clean) >= 3:
                            for lb, tx, *_ in options:
                                if not tx or (lb in used_labels and lb != 'J'):
                                    continue
                                tc = self.clean_text(tx)
                                if len(tc) >= 3 and (ct_clean in tc or tc in ct_clean):
                                    found = lb
                                    break
                        if found:
                            used_labels.add(found)
                            parts.append(f"{found}.{ct}")
                        else:
                            parts.append(ct)
                    self.log("答案: " + " ".join(parts))

                    # 自动点击
                    if self.mode == "auto":
                        if not ADB_PATH:
                            _to_manual(self, "未找到ADB")
                            time.sleep(1)
                            continue
                        if is_first:
                            time.sleep(1)
                            continue
                        for _, x, y in matched_opts:
                            tap_option(int((x - roi_x) * ratio_x),
                                       int((y - roi_y) * ratio_y) + opt_offset)
                            time.sleep(0.5)
                        if next_btn:
                            nx, ny = next_btn
                            tap_option(int((nx - roi_x) * ratio_x),
                                       int((ny - roi_y) * ratio_y) + btn_offset)
                        time.sleep(pre_cool)
                    else:
                        time.sleep(1)

                    self.last_question_key = ""
                    self.first_loop = False

                except Exception as e:
                    _to_manual(self, f"{type(e).__name__}: {e}")
                    wake_screen()
                    time.sleep(1)

        except Exception:
            wake_screen()

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoAnswerApp(root)
    root.mainloop()
