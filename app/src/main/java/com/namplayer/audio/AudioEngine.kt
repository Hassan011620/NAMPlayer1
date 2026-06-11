package com.namplayer.audio

import android.app.*
import android.content.Context
import android.content.Intent
import android.media.*
import android.os.*
import android.util.Log
import com.namplayer.engine.CabIrLoader
import com.namplayer.engine.NoiseGate
import com.namplayer.engine.OnnxInferenceEngine
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlin.math.*

private const val TAG      = "AudioEngine"
private const val CH_ID    = "nam_audio"
private const val NOTIF_ID = 1

enum class EngineState { IDLE, RUNNING, ERROR }

data class AudioStats(
    val latencyMs:    Float   = 0f,
    val cpuPct:       Float   = 0f,
    val inLevelDb:    Float   = -60f,
    val outLevelDb:   Float   = -60f,
    val sampleRate:   Int     = 48000,
    val bufferFrames: Int     = 256,
    val usbActive:    Boolean = false,
    val xruns:        Int     = 0,
    val isRecording:  Boolean = false,
    val recSeconds:   Long    = 0
)

class AudioEngine : Service() {

    inner class EngineBinder : Binder() {
        fun get(): AudioEngine = this@AudioEngine
    }

    private val binder = EngineBinder()
    private val scope  = CoroutineScope(Dispatchers.Default + SupervisorJob())

    // ── DSP chain ─────────────────────────────────────────
    // Input → NoiseGate → NAM (ONNX) → CabIR → Output
    // ──────────────────────────────────────────────────────

    val onnx:     OnnxInferenceEngine by lazy { OnnxInferenceEngine(this) }
    val cabIr:    CabIrLoader   = CabIrLoader()
    val gate:     NoiseGate     = NoiseGate()
    val recorder: AudioRecorder = AudioRecorder(this as Context)

    // ── State ─────────────────────────────────────────────

    private val _state = MutableStateFlow(EngineState.IDLE)
    val state: StateFlow<EngineState> = _state

    private val _stats = MutableStateFlow(AudioStats())
    val stats: StateFlow<AudioStats> = _stats

    // ── Audio objects ─────────────────────────────────────

    private var rec:  AudioRecord? = null
    private var play: AudioTrack?  = null
    private var loop: Job?         = null

    var sampleRate:   Int = 48000; private set
    var bufferFrames: Int = 256;   private set
    var xruns:        Int = 0;     private set

    private var inPeak  = 0f
    private var outPeak = 0f

    // ── Lifecycle ─────────────────────────────────────────

    override fun onBind(intent: Intent?) = binder

    override fun onDestroy() {
        recorder.stop()
        stop()
        scope.cancel()
        onnx.close()
        super.onDestroy()
    }

    // ── Start / Stop ──────────────────────────────────────

    fun start() {
        if (_state.value == EngineState.RUNNING) return
        createChannel()
        startForeground(NOTIF_ID, buildNotif())
        scope.launch {
            runCatching {
                initAudio()
                gate.setSampleRate(sampleRate)
                runLoop()
                _state.value = EngineState.RUNNING
            }.onFailure {
                Log.e(TAG, "start failed", it)
                releaseAudio()
                _state.value = EngineState.ERROR
            }
        }
    }

    fun stop() {
        recorder.stop()
        loop?.cancel(); loop = null
        releaseAudio()
        _state.value = EngineState.IDLE
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    // ── Recording controls ────────────────────────────────

    fun startRecording() = recorder.start(sampleRate)
    fun stopRecording()  = recorder.stop()

    // ── Audio init ────────────────────────────────────────

    private fun initAudio() {
        val am = getSystemService(AUDIO_SERVICE) as AudioManager
        sampleRate   = am.getProperty(AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE)
            ?.toIntOrNull() ?: 48000
        bufferFrames = am.getProperty(AudioManager.PROPERTY_OUTPUT_FRAMES_PER_BUFFER)
            ?.toIntOrNull()?.coerceIn(64, 512) ?: 256

        val recMin = AudioRecord.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        rec = AudioRecord.Builder()
            .setAudioSource(MediaRecorder.AudioSource.UNPROCESSED)
            .setAudioFormat(AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_IN_MONO).build())
            .setBufferSizeInBytes(maxOf(recMin, bufferFrames * 4))
            .build()

        val trkMin = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        play = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build())
            .setAudioFormat(AudioFormat.Builder()
                .setSampleRate(sampleRate)
                .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(maxOf(trkMin, bufferFrames * 4))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setPerformanceMode(AudioTrack.PERFORMANCE_MODE_LOW_LATENCY)
            .build()
    }

    // ── Real-time DSP loop ────────────────────────────────
    //
    // Chain:  input → [NoiseGate] → [NAM/ONNX] → [CabIR] → output
    //                                                ↓
    //                                          [Recorder]
    //
    private fun runLoop() {
        rec?.startRecording()
        play?.play()
        xruns = 0

        val inBuf  = FloatArray(bufferFrames)
        val outBuf = FloatArray(bufferFrames)
        var tick   = 0

        loop = scope.launch(Dispatchers.Default) {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)

            while (isActive) {
                val t0 = System.nanoTime()

                // 1. READ
                val n = rec?.read(inBuf, 0, bufferFrames,
                    AudioRecord.READ_BLOCKING) ?: break
                if (n <= 0) { delay(1); continue }

                // 2. NOISE GATE (pre NAM)
                gate.process(inBuf, n)

                // 3. NAM INFERENCE
                onnx.process(inBuf, outBuf, n)

                // 4. CAB IR CONVOLUTION
                cabIr.process(outBuf, outBuf, n)

                // 5. RECORD (بعد كل المعالجة)
                if (recorder.isRecording) recorder.write(outBuf, n)

                // 6. OUTPUT
                val w = play?.write(outBuf, 0, n, AudioTrack.WRITE_NON_BLOCKING) ?: -1
                if (w < 0) xruns++

                // 7. METERS
                var ip = 0f; var op = 0f
                for (i in 0 until n) {
                    ip = maxOf(ip, abs(inBuf[i]))
                    op = maxOf(op, abs(outBuf[i]))
                }
                inPeak  = maxOf(ip,  inPeak  * 0.96f)
                outPeak = maxOf(op,  outPeak * 0.96f)

                // 8. STATS (every 40 ticks)
                if (++tick >= 40) {
                    tick = 0
                    val cpuNs = System.nanoTime() - t0
                    val bufMs = bufferFrames * 1000f / sampleRate
                    _stats.value = AudioStats(
                        latencyMs    = bufferFrames * 2f * 1000f / sampleRate,
                        cpuPct       = (cpuNs / 1_000_000f / bufMs * 100f).coerceIn(0f, 100f),
                        inLevelDb    = if (inPeak  > 0f) 20f * log10(inPeak)  else -60f,
                        outLevelDb   = if (outPeak > 0f) 20f * log10(outPeak) else -60f,
                        sampleRate   = sampleRate,
                        bufferFrames = bufferFrames,
                        xruns        = xruns,
                        isRecording  = recorder.isRecording,
                        recSeconds   = recorder.durationSeconds
                    )
                    inPeak  *= 0.5f
                    outPeak *= 0.5f
                }
            }
        }
    }

    // ── Cleanup ───────────────────────────────────────────

    private fun releaseAudio() {
        runCatching { rec?.stop();  rec?.release()  }
        runCatching { play?.stop(); play?.release() }
        rec = null; play = null
    }

    // ── Notification ──────────────────────────────────────

    private fun createChannel() {
        getSystemService(NotificationManager::class.java)
            .createNotificationChannel(
                NotificationChannel(CH_ID, "NAM Audio",
                    NotificationManager.IMPORTANCE_LOW).apply {
                    setShowBadge(false); setSound(null, null)
                })
    }

    private fun buildNotif() = Notification.Builder(this, CH_ID)
        .setContentTitle("NAM Player")
        .setContentText("معالجة الصوت نشطة")
        .setSmallIcon(android.R.drawable.ic_media_play)
        .setOngoing(true)
        .build()
}
