package com.namplayer.audio

import android.content.Context
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*

private const val TAG = "AudioRecorder"

// ─────────────────────────────────────────────────────────────
// AudioRecorder
// يسجّل الصوت المعالج (بعد NAM + Cab IR) إلى ملف WAV
// ─────────────────────────────────────────────────────────────

class AudioRecorder(private val ctx: Context) {

    enum class RecState { IDLE, RECORDING }

    private val _state = MutableStateFlow(RecState.IDLE)
    val state: StateFlow<RecState> = _state

    private var outStream: FileOutputStream? = null
    private var wavFile:   File? = null
    private var sampleRate = 48000
    private var samplesWritten = 0L

    val isRecording get() = _state.value == RecState.RECORDING

    // ── Start recording ───────────────────────────────────

    fun start(sr: Int): Boolean {
        if (isRecording) return false
        sampleRate = sr
        samplesWritten = 0

        return runCatching {
            val dir = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC),
                "NAMPlayer"
            ).also { it.mkdirs() }

            val ts   = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            wavFile  = File(dir, "nam_rec_$ts.wav")
            outStream = FileOutputStream(wavFile!!)

            // اكتب WAV header مؤقت (سيُحدَّث عند الإيقاف)
            writeWavHeader(outStream!!, 0, sr)

            _state.value = RecState.RECORDING
            Log.i(TAG, "Recording started: ${wavFile!!.name}")
            true
        }.onFailure {
            Log.e(TAG, "Start failed: ${it.message}")
        }.getOrDefault(false)
    }

    // ── Write audio chunk ─────────────────────────────────

    fun write(buf: FloatArray, frames: Int) {
        if (!isRecording) return
        val stream = outStream ?: return

        runCatching {
            val bytes = ByteBuffer.allocate(frames * 2)
                .order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until frames) {
                val s = (buf[i].coerceIn(-1f, 1f) * 32767f).toInt().toShort()
                bytes.putShort(s)
            }
            stream.write(bytes.array())
            samplesWritten += frames
        }
    }

    // ── Stop and finalize WAV ─────────────────────────────

    fun stop(): File? {
        if (!isRecording) return null
        val file = wavFile ?: return null

        runCatching {
            outStream?.flush()
            outStream?.close()
            outStream = null

            // أعد كتابة الـ WAV header بالحجم الصحيح
            val raf = RandomAccessFile(file, "rw")
            writeWavHeaderRaf(raf, samplesWritten, sampleRate)
            raf.close()

            Log.i(TAG, "Recording saved: ${file.name} " +
                  "(${samplesWritten} samples, ${samplesWritten / sampleRate}s)")
        }.onFailure {
            Log.e(TAG, "Stop failed: ${it.message}")
        }

        _state.value = RecState.IDLE
        val saved = file
        wavFile = null
        return saved
    }

    val durationSeconds: Long get() =
        if (sampleRate > 0) samplesWritten / sampleRate else 0

    val filePath: String? get() = wavFile?.absolutePath

    // ── WAV header writers ────────────────────────────────

    private fun writeWavHeader(stream: OutputStream, samples: Long, sr: Int) {
        val dataSize   = (samples * 2).toInt()
        val totalSize  = dataSize + 36

        val buf = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray())
        buf.putInt(totalSize)
        buf.put("WAVE".toByteArray())
        buf.put("fmt ".toByteArray())
        buf.putInt(16)           // chunk size
        buf.putShort(1)          // PCM
        buf.putShort(1)          // mono
        buf.putInt(sr)           // sample rate
        buf.putInt(sr * 2)       // byte rate
        buf.putShort(2)          // block align
        buf.putShort(16)         // bits per sample
        buf.put("data".toByteArray())
        buf.putInt(dataSize)
        stream.write(buf.array())
    }

    private fun writeWavHeaderRaf(raf: RandomAccessFile, samples: Long, sr: Int) {
        val dataSize  = (samples * 2)
        val totalSize = dataSize + 36

        raf.seek(0)
        raf.write("RIFF".toByteArray())
        raf.write(intToLeBytes(totalSize.toInt()))
        raf.write("WAVE".toByteArray())
        raf.write("fmt ".toByteArray())
        raf.write(intToLeBytes(16))
        raf.write(shortToLeBytes(1))
        raf.write(shortToLeBytes(1))
        raf.write(intToLeBytes(sr))
        raf.write(intToLeBytes(sr * 2))
        raf.write(shortToLeBytes(2))
        raf.write(shortToLeBytes(16))
        raf.write("data".toByteArray())
        raf.write(intToLeBytes(dataSize.toInt()))
    }

    private fun intToLeBytes(v: Int)   = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array()
    private fun shortToLeBytes(v: Int) = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(v.toShort()).array()
}
