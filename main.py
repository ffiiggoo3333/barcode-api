from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.utils import get_color_from_hex
from plyer import flashlight, vibrator, request_permissions
from plyer.facades.vibrator import Vibrator
from plyer.facades.flashlight import Flashlight
from plyer.utils import platform

import cv2
import numpy as np
from ultralytics import YOLO
from pyzbar.pyzbar import decode
from plyer import filechooser
import os

class BarcodeApp(App):
    def build(self):
        # Request necessary permissions for Android
        if platform == 'android':
            request_permissions([
                'android.permission.CAMERA',
                'android.permission.WRITE_EXTERNAL_STORAGE',
                'android.permission.FLASHLIGHT',
                'android.permission.VIBRATE'
            ])

        # Set background color
        Window.clearcolor = get_color_from_hex("#f0f0f0")

        try:
            # تحميل نموذج YOLOv8s
self.model = YOLO("YOLOV8s_Barcode_Detection.pt")

# تقليل FPS لتخفيف الحمل على الهاتف
self.event = Clock.schedule_interval(self.update, 1.0 / 10.0)  # 10fps

        except FileNotFoundError:
            self.label = Label(
                text="خطأ: لم يتم العثور على ملف النموذج. يرجى التأكد من وجوده في مجلد التطبيق.",
                size_hint_y=None,
                height=dp(44),
                font_size=dp(18),
                bold=True,
                color=get_color_from_hex("#FF0000")
            )
            return self.label
        
        # Check if camera is available before initializing
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.label = Label(
                text="خطأ: تعذر الوصول إلى الكاميرا. يرجى التحقق من الأذونات.",
                size_hint_y=None,
                height=dp(44),
                font_size=dp(18),
                bold=True,
                color=get_color_from_hex("#FF0000")
            )
            return self.label

        self.flashlight_on = False
        self.scanned_count = 0
        
        self.seen_barcodes = set()
        self.output_folder = None
        self.output_file = None
        
        # Create a default folder
        default_folder = "Barcode_Scans"
        self.output_folder = os.path.join(self.user_data_dir, default_folder)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.output_file = os.path.join(self.output_folder, "detected_barcodes.txt")
        
        self.img_widget = Image(
            allow_stretch=True, 
            keep_ratio=True,
            size_hint=(1, 0.7)
        )
        
        self.label = Label(
            text=f"مجلد المخرجات الافتراضي: {self.output_folder}",
            size_hint_y=None,
            height=dp(44),
            font_size=dp(18),
            bold=True,
            color=get_color_from_hex("#333333")
        )

        self.status_label = Label(
            text="عدد الباركودات: 0",
            size_hint_y=None,
            height=dp(44),
            font_size=dp(16),
            color=get_color_from_hex("#666666")
        )

        self.start_btn = Button(
            text="ابدأ",
            on_press=self.toggle,
            size_hint_y=None,
            height=dp(50),
            background_normal='',
            background_color=get_color_from_hex("#4CAF50"),
            color=get_color_from_hex("#FFFFFF"),
            font_size=dp(20)
        )
        self.choose_btn = Button(
            text="اختر مجلد",
            on_press=lambda x: Clock.schedule_once(self.choose_folder, 0),
            size_hint_y=None,
            height=dp(50),
            background_normal='',
            background_color=get_color_from_hex("#2196F3"),
            color=get_color_from_hex("#FFFFFF"),
            font_size=dp(20)
        )
        
        self.flash_btn = Button(
            text="تشغيل الفلاش",
            on_press=self.toggle_flash,
            size_hint_y=None,
            height=dp(50),
            background_normal='',
            background_color=get_color_from_hex("#FF9800"),
            color=get_color_from_hex("#FFFFFF"),
            font_size=dp(20)
        )

        layout = BoxLayout(
            orientation='vertical',
            padding=dp(20),
            spacing=dp(10)
        )
        
        button_layout = BoxLayout(
            orientation='horizontal',
            spacing=dp(10),
            size_hint_y=None,
            height=dp(50)
        )

        button_layout.add_widget(self.start_btn)
        button_layout.add_widget(self.choose_btn)
        button_layout.add_widget(self.flash_btn)

        layout.add_widget(self.img_widget)
        layout.add_widget(self.label)
        layout.add_widget(self.status_label)
        layout.add_widget(button_layout)

        # Load previously seen barcodes from the file if it exists
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.seen_barcodes.add(line.strip())
            self.scanned_count = len(self.seen_barcodes)
            self.status_label.text = f"عدد الباركودات: {self.scanned_count}"

        # Dictionary to track barcode box colors
        self.barcode_colors = {}

        return layout
    
    def toggle_flash(self, instance):
        if not self.flashlight_on:
            try:
                flashlight.on()
                self.flashlight_on = True
                self.flash_btn.text = "إيقاف الفلاش"
            except NotImplementedError:
                self.label.text = "خطأ: الفلاش غير مدعوم على هذا الجهاز."
        else:
            try:
                flashlight.off()
                self.flashlight_on = False
                self.flash_btn.text = "تشغيل الفلاش"
            except NotImplementedError:
                self.label.text = "خطأ: الفلاش غير مدعوم على هذا الجهاز."

    def choose_folder(self, dt):
        try:
            filechooser.choose_dir(on_selection=self.folder_selected)
        except Exception as e:
            self.label.text = f"خطأ في اختيار المجلد: {e}"

    def folder_selected(self, selection):
        if selection:
            self.output_folder = selection[0]
            self.output_file = os.path.join(self.output_folder, "detected_barcodes.txt")
            self.label.text = f"مجلد المخرجات: {self.output_folder}"
            
            # Load previously seen barcodes from the file
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.seen_barcodes.add(line.strip())
                self.scanned_count = len(self.seen_barcodes)
                self.status_label.text = f"عدد الباركودات: {self.scanned_count}"
            
    def toggle(self, instance):
        if not self.running:
            self.event = Clock.schedule_interval(self.update, 1.0 / 30.0)
            self.start_btn.text = "إيقاف"
            self.start_btn.background_color = get_color_from_hex("#F44336")
            self.running = True
        else:
            if self.event:
                self.event.cancel()
            self.start_btn.text = "ابدأ"
            self.start_btn.background_color = get_color_from_hex("#4CAF50")
            self.running = False

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            self.label.text = "خطأ: تعذر القراءة من الكاميرا."
            self.toggle(None) # Stop the app loop
            return

        results = self.model(frame, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            
            barcode_roi = frame[y1:y2, x1:x2]
            decoded_objects = decode(barcode_roi)
            
            label_text = ""
            box_color = (0, 255, 0) # أخضر افتراضي

            if decoded_objects:
                for obj in decoded_objects:
                    data = obj.data.decode('utf-8')

                    if data not in self.seen_barcodes:
                        # Barcode is new, change color to blue and save it
                        self.seen_barcodes.add(data)
                        self.scanned_count += 1
                        self.label.text = f"باركود جديد: {data}"
                        self.status_label.text = f"عدد الباركودات: {self.scanned_count}"
                        
                        box_color = (255, 0, 0) # أزرق
                        self.barcode_colors[data] = box_color # Store color in dictionary
                        
                        label_text = f"جديد: {data}"
                        
                        try:
                            vibrator.vibrate(0.5) # Vibrate for 0.5 seconds
                        except NotImplementedError:
                            self.label.text = "خطأ: الاهتزاز غير مدعوم على هذا الجهاز."
                            
                        if self.output_file:
                            with open(self.output_file, 'a', encoding='utf-8') as f:
                                f.write(data + '\n')
                        
                        # Schedule color change to green after 1 second
                        Clock.schedule_once(lambda dt, d=data: self.change_color_to_green(d), 1)

                    else:
                        # Barcode is already seen, use its stored color (green)
                        box_color = self.barcode_colors.get(data, (0, 255, 0)) # Green
                        label_text = f"تمت رؤيته: {data}"
            else:
                # If no barcode detected, show model confidence
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label_text = f"{self.model.names[cls]} {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img_widget.texture = texture

    def change_color_to_green(self, data):
        """Changes the color of a barcode's box to green."""
        self.barcode_colors[data] = (0, 255, 0) # Green

    def on_stop(self):
        self.capture.release()
        if self.flashlight_on:
            try:
                flashlight.off()
            except NotImplementedError:
                pass

if __name__ == "__main__":
    BarcodeApp().run()
