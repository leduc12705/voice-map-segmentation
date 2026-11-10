#!/usr/bin/env python3
"""Nhan dang giong noi -> topic /voice_text (std_msgs/String).

mode:=vosk  : micro + Vosk (pip install vosk sounddevice; tai model tieng Viet vosk-model-vn-0.4)
mode:=whisper : micro + faster-whisper (pip install faster-whisper), nhan dang tieng Viet tot hon Vosk
mode:=stdin : go cau lenh bang ban phim (dung khi VM khong co micro / de test)
"""
import json
import queue
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def build_phrases(zones_file):
    """Tao danh sach cau lenh hop le (tieng Anh) tu zones.yaml de Vosk chi nhan dang trong tap nay."""
    import yaml
    zones = yaml.safe_load(open(zones_file, encoding="utf-8"))["zones"]
    words = []
    for z in zones:
        for a in z.get("aliases", []):
            if a.isascii() and not a.startswith(("khu", "vung", "zone")) and a not in words:
                words.append(a)
    phrases = ["stop", "[unk]", "go to [unk]", "go to the [unk]"]
    for w in words:
        phrases += [f"go to {w}", f"go to the {w}"]
    for a in words:
        for b in words:
            if a != b:
                phrases += [f"move from {a} to {b}", f"go from {a} to {b}"]
    return phrases


class SpeechNode(Node):
    def __init__(self):
        super().__init__("speech_node")
        self.declare_parameter("mode", "stdin")
        self.declare_parameter("model_path", "")
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("device", -1)  # -1 = micro mac dinh
        self.declare_parameter("language", "vi")
        self.declare_parameter("zones_file", "")
        self.declare_parameter("whisper_model", "small")   # tiny | base | small | medium
        self.declare_parameter("energy_threshold", 400)    # muc 'amluong' de coi la dang noi
        self.declare_parameter("silence_s", 0.8)           # im lang bao lau thi ket thuc cau
        self.pub = self.create_publisher(String, "voice_text", 10)
        mode = self.get_parameter("mode").value
        target = {"vosk": self._run_vosk, "whisper": self._run_whisper}.get(mode, self._run_stdin)
        threading.Thread(target=target, daemon=True).start()

    def _publish(self, text):
        self.get_logger().info(f"Nghe duoc: '{text}'")
        self.pub.publish(String(data=text))

    def _run_stdin(self):
        self.get_logger().info("Che do ban phim: go cau lenh roi Enter")
        while rclpy.ok():
            try:
                text = input("lenh> ").strip()
            except EOFError:
                return
            if text:
                self._publish(text)

    def _run_vosk(self):
        import sounddevice as sd
        from vosk import KaldiRecognizer, Model
        path = self.get_parameter("model_path").value
        sr = int(self.get_parameter("sample_rate").value)
        dev = int(self.get_parameter("device").value)
        model = Model(path)
        zf = self.get_parameter("zones_file").value
        if zf:
            phrases = build_phrases(zf)
            self.get_logger().info(f"Vosk chi nhan {len(phrases)} cau lenh hop le")
            rec = KaldiRecognizer(model, sr, json.dumps(phrases))
        else:
            rec = KaldiRecognizer(model, sr)
        q = queue.Queue()

        def cb(indata, frames, t, status):
            q.put(bytes(indata))

        self.get_logger().info("Dang nghe micro ... (hay noi cau lenh)")
        with sd.RawInputStream(samplerate=sr, blocksize=8000, dtype="int16", channels=1,
                               device=None if dev < 0 else dev, callback=cb):
            while rclpy.ok():
                if rec.AcceptWaveform(q.get()):
                    text = json.loads(rec.Result()).get("text", "").strip()
                    if text and text != "[unk]":
                        self._publish(text)


    def _run_whisper(self):
        from collections import deque

        import numpy as np
        import sounddevice as sd
        from faster_whisper import WhisperModel

        sr = 16000
        blk = int(0.1 * sr)  # 0.1 s moi khoi
        thr = float(self.get_parameter("energy_threshold").value)
        silence_blocks = int(float(self.get_parameter("silence_s").value) / 0.1)
        dev = int(self.get_parameter("device").value)
        size = self.get_parameter("whisper_model").value
        self.get_logger().info(f"Dang nap Whisper '{size}' (lan dau se tai model) ...")
        model = WhisperModel(size, device="cpu", compute_type="int8")
        prompt = ("Đi đến khu vực A. Di chuyển từ khu vực A đến khu vực B. "
                  "Đi đến khu vực C. Đi đến khu vực D. Đi đến phòng thí nghiệm.")
        q = queue.Queue()

        def cb(indata, frames, t, status):
            q.put(indata[:, 0].copy())

        def transcribe(chunks):
            audio = np.concatenate(chunks).astype(np.float32) / 32768.0
            if len(audio) < 0.4 * sr:
                return
            segs, _ = model.transcribe(audio, language=self.get_parameter("language").value, beam_size=5,
                                       condition_on_previous_text=False)
            text = " ".join(s.text for s in segs).strip()
            if text:
                self._publish(text)

        self.get_logger().info("Dang nghe micro ... (noi xong, im lang ~1 giay)")
        preroll, buf, speaking, quiet = deque(maxlen=3), [], False, 0
        with sd.InputStream(samplerate=sr, blocksize=blk, dtype="int16", channels=1,
                            device=None if dev < 0 else dev, callback=cb):
            while rclpy.ok():
                chunk = q.get()
                loud = np.abs(chunk).mean() > thr
                if not speaking:
                    preroll.append(chunk)
                    if loud:
                        speaking, quiet, buf = True, 0, list(preroll)
                    continue
                buf.append(chunk)
                quiet = 0 if loud else quiet + 1
                if quiet >= silence_blocks or len(buf) >= 80:  # toi da 8 giay
                    transcribe(buf)
                    speaking, buf = False, []
                    preroll.clear()


def main():
    rclpy.init()
    node = SpeechNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
