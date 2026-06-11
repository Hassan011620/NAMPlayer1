package com.namplayer.ui

import android.app.Application
import android.content.*
import android.net.Uri
import android.os.IBinder
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.namplayer.audio.AudioEngine
import com.namplayer.audio.AudioStats
import com.namplayer.audio.EngineState
import com.namplayer.converter.NamModel
import com.namplayer.converter.NamParser
import com.namplayer.usb.UsbAudioDev
import com.namplayer.usb.UsbAudioManager
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val ctx = app.applicationContext
    val usb = UsbAudioManager(ctx)

    // ── Service ───────────────────────────────────────────

    private var svc: AudioEngine? = null
    private val conn = object : ServiceConnection {
        override fun onServiceConnected(n: ComponentName, b: IBinder) {
            svc = (b as AudioEngine.EngineBinder).get()
            observeService()
        }
        override fun onServiceDisconnected(n: ComponentName) { svc = null }
    }

    // ── State flows ───────────────────────────────────────

    private val _engineState = MutableStateFlow(EngineState.IDLE)
    val engineState: StateFlow<EngineState> = _engineState

    private val _stats = MutableStateFlow(AudioStats())
    val stats: StateFlow<AudioStats> = _stats

    private val _model   = MutableStateFlow<NamModel?>(null)
    val model: StateFlow<NamModel?> = _model

    private val _cabName = MutableStateFlow<String?>(null)
    val cabName: StateFlow<String?> = _cabName

    private val _gateOn  = MutableStateFlow(false)
    val gateEnabled: StateFlow<Boolean> = _gateOn

    private val _gateThresh = MutableStateFlow(-60f)
    val gateThreshDb: StateFlow<Float> = _gateThresh

    private val _loading = MutableStateFlow(false)
    val loading: StateFlow<Boolean> = _loading

    val usbDevices: StateFlow<List<UsbAudioDev>> = usb.devices
    val activeUsb:  StateFlow<UsbAudioDev?>      = usb.active

    private val _msg = MutableSharedFlow<String>()
    val msg: SharedFlow<String> = _msg

    val isRunning get() = _engineState.value == EngineState.RUNNING

    init {
        ctx.bindService(Intent(ctx, AudioEngine::class.java), conn, Context.BIND_AUTO_CREATE)
        usb.scan()
    }

    private fun observeService() {
        val s = svc ?: return
        viewModelScope.launch { s.state.collect { _engineState.value = it } }
        viewModelScope.launch { s.stats.collect { _stats.value = it } }
    }

    // ── Audio controls ────────────────────────────────────

    fun toggle() { if (isRunning) svc?.stop() else svc?.start() }
    fun setInputGain(db: Float)  { svc?.onnx?.setInputGainDb(db) }
    fun setOutputGain(db: Float) { svc?.onnx?.setOutputGainDb(db) }

    // ── Load .nam ─────────────────────────────────────────

    fun loadNam(uri: Uri) = viewModelScope.launch {
        _loading.value = true
        runCatching {
            val stream = ctx.contentResolver.openInputStream(uri) ?: error("Cannot open")
            val json   = NamParser.parse(stream).getOrThrow()
            val mdl    = NamParser.build(json).getOrThrow()
            val ok     = svc?.onnx?.load(mdl) ?: false
            if (ok) { _model.value = mdl; _msg.emit("✅ ${mdl.displayName}") }
            else    _msg.emit("❌ فشل التحميل")
        }.onFailure { _msg.emit("❌ ${it.message}") }
        _loading.value = false
    }

    // ── Load Cab IR (.wav) ────────────────────────────────

    fun loadCab(uri: Uri) = viewModelScope.launch {
        _loading.value = true
        runCatching {
            val name   = getFileName(uri)
            val stream = ctx.contentResolver.openInputStream(uri) ?: error("Cannot open")
            val ok     = svc?.cabIr?.load(stream, name) ?: false
            if (ok) { _cabName.value = name; _msg.emit("🔊 Cabinet: $name") }
            else    _msg.emit("❌ ملف WAV غير صالح")
        }.onFailure { _msg.emit("❌ ${it.message}") }
        _loading.value = false
    }

    fun clearCab() {
        svc?.cabIr?.unload()
        _cabName.value = null
        viewModelScope.launch { _msg.emit("🔊 Cabinet: off") }
    }

    fun setCabEnabled(on: Boolean) { svc?.cabIr?.let { it.enabled = on } }

    // ── Noise Gate ────────────────────────────────────────

    fun setGateEnabled(on: Boolean) {
        _gateOn.value = on
        svc?.gate?.enabled = on
    }

    fun setGateThreshold(db: Float) {
        _gateThresh.value = db
        svc?.gate?.thresholdDb = db
    }

    // ── Recording ─────────────────────────────────────────

    fun toggleRecord() {
        val s = svc ?: return
        if (s.recorder.isRecording) {
            val file = s.stopRecording()
            viewModelScope.launch {
                _msg.emit(if (file != null) "💾 محفوظ: ${file.name}" else "❌ خطأ في الحفظ")
            }
        } else {
            val ok = s.startRecording()
            viewModelScope.launch { _msg.emit(if (ok) "⏺ تسجيل..." else "❌ لا يمكن التسجيل") }
        }
    }

    // ── USB ───────────────────────────────────────────────

    fun onUsbAttached() { usb.scan() }
    fun onUsbDetached() { usb.disconnect(); usb.scan() }
    fun selectUsb(d: UsbAudioDev) {
        usb.request(d)
        viewModelScope.launch { _msg.emit("🔌 ${d.label}") }
    }

    // ── Helpers ───────────────────────────────────────────

    private fun getFileName(uri: Uri): String = runCatching {
        ctx.contentResolver.query(uri, null, null, null, null)?.use { c ->
            val i = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            c.moveToFirst(); c.getString(i)
        }
    }.getOrNull() ?: uri.lastPathSegment ?: "unknown"

    override fun onCleared() {
        super.onCleared()
        runCatching { ctx.unbindService(conn) }
    }
}
