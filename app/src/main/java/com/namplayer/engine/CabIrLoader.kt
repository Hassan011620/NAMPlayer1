package com.namplayer.engine

import android.content.Context
import android.net.Uri
import android.util.Log
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

private const val TAG = "CabIR"

// ─────────────────────────────────────────────────────────────
// Cabinet IR Loader + Fast Convolver
// يحمّل ملف .wav ويطبّق الـ convolution على الصوت
// يستخدم Overlap-Add FFT لأداء مرتفع
// ─────────────────────────────────────────────────────────────

class CabIrLoader {

    var isLoaded: Boolean = false
        private set

    var irName: String = ""
        private set

    // IR kernel (مقطوع إلى 2048 sample للأداء)
    private var kernel: FloatArray = FloatArray(0)
    private var kernelLen: Int = 0

    // Overlap-Add state
    private var overlap: FloatArray = FloatArray(0)

    // Enable/disable bypass
    var enabled: Boolean = true

    // ── Load .wav IR file ──────────────────────────────────

    fun load(stream: InputStream, name: String): Boolean {
        return runCatching {
            val wav = parseWav(stream) ?: error("Invalid WAV file")

            // resample إذا لزم — نستخدم أول 2048 sample فقط
            val maxLen = 2048
            kernel = if (wav.size > maxLen) wav.copyOfRange(0, maxLen) else wav
            kernelLen = kernel.size
            overlap = FloatArray(kernelLen)
            irName = name
            isLoaded = true
            Log.i(TAG, "IR loaded: $name, len=$kernelLen")
            true
        }.onFailure {
            Log.e(TAG, "IR load failed: ${it.message}")
            isLoaded = false
        }.getOrDefault(false)
    }

    fun load(ctx: Context, uri: Uri, name: String): Boolean {
        val stream = ctx.contentResolver.openInputStream(uri) ?: return false
        return load(stream, name)
    }

    fun unload() {
        kernel = FloatArray(0)
        overlap = FloatArray(0)
        kernelLen = 0
        isLoaded = false
        irName = ""
    }

    // ── Process: Overlap-Add convolution ──────────────────

    fun process(input: FloatArray, output: FloatArray, frames: Int) {
        if (!isLoaded || !enabled || kernelLen == 0) {
            input.copyInto(output, 0, 0, frames)
            return
        }

        // Direct convolution للـ buffers الصغيرة (< 512)
        // Overlap-Add لو أكبر
        if (frames <= 128) {
            directConvolve(input, output, frames)
        } else {
            overlapAdd(input, output, frames)
        }
    }

    // ── Direct convolution (low latency for small buffers) ─

    private fun directConvolve(input: FloatArray, output: FloatArray, frames: Int) {
        val kLen = minOf(kernelLen, 256) // limit for speed
        for (n in 0 until frames) {
            var sum = 0f
            val limit = minOf(kLen, n + 1)
            for (k in 0 until limit) {
                sum += kernel[k] * (if (n - k < frames) input[n - k] else overlap[kernelLen - (k - n) - 1])
            }
            output[n] = sum
        }
        // save tail for next block
        val copyLen = minOf(frames, kernelLen)
        input.copyInto(overlap, kernelLen - copyLen, 0, copyLen)
    }

    // ── Overlap-Add (efficient for larger buffers) ────────

    private fun overlapAdd(input: FloatArray, output: FloatArray, frames: Int) {
        val fftSize = nextPow2(frames + kernelLen - 1)

        // Extend input with zeros
        val x = FloatArray(fftSize).also { input.copyInto(it, 0, 0, frames) }
        val h = FloatArray(fftSize).also { kernel.copyInto(it, 0, 0, kernelLen) }

        // FFT-based linear convolution
        val xR = x.copyOf(); val xI = FloatArray(fftSize)
        val hR = h.copyOf(); val hI = FloatArray(fftSize)
        fft(xR, xI, false)
        fft(hR, hI, false)

        // Complex multiply
        val yR = FloatArray(fftSize)
        val yI = FloatArray(fftSize)
        for (i in 0 until fftSize) {
            yR[i] = xR[i] * hR[i] - xI[i] * hI[i]
            yI[i] = xR[i] * hI[i] + xI[i] * hR[i]
        }

        // IFFT
        fft(yR, yI, true)

        // Add overlap from previous block
        for (i in 0 until frames) {
            output[i] = yR[i] + (if (i < overlap.size) overlap[i] else 0f)
        }

        // Save tail
        val tailLen = fftSize - frames
        overlap = FloatArray(maxOf(tailLen, kernelLen))
        for (i in 0 until tailLen) {
            overlap[i] = if (frames + i < fftSize) yR[frames + i] else 0f
        }
    }

    // ── WAV Parser ────────────────────────────────────────

    private fun parseWav(stream: InputStream): FloatArray? {
        val bytes = stream.readBytes()
        if (bytes.size < 44) return null

        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // RIFF header check
        val riff = String(bytes, 0, 4)
        if (riff != "RIFF") return null
        val wave = String(bytes, 8, 4)
        if (wave != "WAVE") return null

        bb.position(12)
        var audioData: ByteArray? = null
        var numChannels = 1
        var bitsPerSample = 16
        var sampleRate = 44100

        // Parse chunks
        while (bb.remaining() >= 8) {
            val chunkId   = String(bytes, bb.position(), 4)
            bb.position(bb.position() + 4)
            val chunkSize = bb.int

            when (chunkId) {
                "fmt " -> {
                    bb.short // audioFormat
                    numChannels  = bb.short.toInt()
                    sampleRate   = bb.int
                    bb.int   // byteRate
                    bb.short // blockAlign
                    bitsPerSample = bb.short.toInt()
                    val remaining = chunkSize - 16
                    if (remaining > 0) bb.position(bb.position() + remaining)
                }
                "data" -> {
                    audioData = ByteArray(chunkSize)
                    bb.get(audioData)
                }
                else -> {
                    val skip = minOf(chunkSize, bb.remaining())
                    bb.position(bb.position() + skip)
                }
            }
        }

        val data = audioData ?: return null
        val dataBuf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)

        // Convert to float mono
        val totalSamples = data.size / (bitsPerSample / 8)
        val samplesPerChannel = totalSamples / numChannels
        val result = FloatArray(samplesPerChannel)

        for (i in 0 until samplesPerChannel) {
            var sum = 0f
            for (ch in 0 until numChannels) {
                sum += when (bitsPerSample) {
                    16   -> dataBuf.short / 32768f
                    24   -> {
                        val b0 = dataBuf.get().toInt() and 0xFF
                        val b1 = dataBuf.get().toInt() and 0xFF
                        val b2 = dataBuf.get().toInt()
                        ((b2 shl 16) or (b1 shl 8) or b0) / 8388608f
                    }
                    32   -> dataBuf.float
                    else -> dataBuf.short / 32768f
                }
            }
            result[i] = sum / numChannels
        }

        Log.d(TAG, "WAV: ${samplesPerChannel} samples, ${sampleRate}Hz, ${bitsPerSample}bit, ${numChannels}ch")
        return result
    }

    // ── Helpers ───────────────────────────────────────────

    private fun nextPow2(n: Int): Int {
        var p = 1; while (p < n) p = p shl 1; return p
    }

    // Cooley-Tukey FFT in-place
    private fun fft(re: FloatArray, im: FloatArray, inverse: Boolean) {
        val n = re.size
        // Bit reversal
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i];  im[i] = im[j];  im[j] = t
            }
        }
        // Butterfly
        var len = 2
        while (len <= n) {
            val ang = 2.0 * PI / len * (if (inverse) -1 else 1)
            val wRe = cos(ang).toFloat()
            val wIm = sin(ang).toFloat()
            var i = 0
            while (i < n) {
                var curRe = 1f; var curIm = 0f
                for (jj in 0 until len / 2) {
                    val uRe = re[i + jj]
                    val uIm = im[i + jj]
                    val vRe = re[i + jj + len/2] * curRe - im[i + jj + len/2] * curIm
                    val vIm = re[i + jj + len/2] * curIm + im[i + jj + len/2] * curRe
                    re[i + jj]        = uRe + vRe
                    im[i + jj]        = uIm + vIm
                    re[i + jj + len/2] = uRe - vRe
                    im[i + jj + len/2] = uIm - vIm
                    val newRe = curRe * wRe - curIm * wIm
                    curIm = curRe * wIm + curIm * wRe
                    curRe = newRe
                }
                i += len
            }
            len = len shl 1
        }
        if (inverse) { val s = 1f / n; for (i in re.indices) { re[i] *= s; im[i] *= s } }
    }
}
