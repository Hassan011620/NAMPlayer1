# NAM Player v2 — Android

تطبيق أندرويد خفيف لتشغيل ملفات `.nam` في الوقت الحقيقي
مع دعم USB Audio Interface.

---

## البنية التقنية

```
NAMPlayer2/
├── converter/
│   └── NamParser.kt         ← يقرأ .nam JSON → NamModel
├── engine/
│   ├── OnnxInferenceEngine  ← ONNX Runtime + fallback native
│   └── OnnxBuilder.kt       ← يبني ONNX bytes من الأوزان
├── audio/
│   └── AudioEngine.kt       ← Foreground Service + AudioRecord/Track
├── usb/
│   └── UsbAudioManager.kt   ← كشف USB Audio devices
└── ui/
    ├── MainActivity.kt
    └── MainViewModel.kt
```

### لماذا ONNX Runtime؟
- **10x أسرع** من Kotlin خالص للـ LSTM
- يستخدم **NNAPI** (Neural Networks API) إذا كان متاحاً
- دعم **ARM NEON** SIMD تلقائي
- CPU load < 20% على هاتف متوسط
- Latency < 10ms مع buffer size صغير

---

## متطلبات البناء

| الأداة | الإصدار |
|--------|---------|
| Android Studio | Hedgehog 2023.1.1+ |
| JDK | 17 |
| Android SDK | API 26+ |
| Kotlin | 1.9.22 |
| ONNX Runtime | 1.17.0 (يُحمَّل تلقائياً) |

---

## خطوات البناء

### 1 — افتح المشروع
```
Android Studio → File → Open → NAMPlayer2
```

### 2 — Sync
```
File → Sync Project with Gradle Files
```
انتظر حتى يُحمَّل `onnxruntime-android:1.17.0` (~15 MB)

### 3 — Build APK
```
Build → Build Bundle(s)/APK(s) → Build APK(s)
```

الـ APK في:
```
app/build/outputs/apk/debug/app-debug.apk
```

### 4 — نقل إلى الهاتف
- عبر USB: `adb install app-debug.apk`
- أو انسخ الملف وافتحه من مدير الملفات

---

## الاستخدام

### تشغيل موديل .nam
1. اضغط **📂 .nam** واختر الملف
2. اضغط **▶** (كبير في المنتصف)
3. العب على الغيتار!

### مسارات .nam المدعومة
```
/sdcard/NAM/
/sdcard/Downloads/
مدير الملفات أي مكان
```

### USB Audio Interface
```
الهاتف ──[USB-C OTG]── Interface ──[Jack 6.35mm]── غيتار
```
1. وصّل الـ interface
2. اضغط **🔌 USB** → اختر الجهاز
3. امنح الصلاحية

#### أجهزة مدعومة
| الجهاز | UAC | SR |
|--------|-----|----|
| Focusrite Scarlett 2i2 | UAC2 🟢 | 96kHz |
| Behringer UMC22 | UAC1 🟡 | 48kHz |
| Line 6 HX Stomp | UAC2 🟢 | 96kHz |
| PreSonus AudioBox | UAC1 🟡 | 48kHz |
| IK AXE I/O | UAC2 🟢 | 96kHz |
| Roland UA-55 | UAC1 🟡 | 48kHz |

---

## معلومات تقنية

| المعامل | القيمة |
|---------|--------|
| Audio Source | UNPROCESSED (بدون Android EQ) |
| Format | Float32 PCM |
| Mode | LOW_LATENCY |
| Sample Rate | Native (44.1k أو 48k) |
| Buffer | Native frames per buffer |
| Thread Priority | URGENT_AUDIO |

---

## متطلبات الهاتف

- Android 8.0+ (API 26)
- RAM: 1GB كافية
- معالج: ARM64 (arm64-v8a) مُفضَّل
- لـ USB: يجب أن يدعم الهاتف USB Host (OTG)
