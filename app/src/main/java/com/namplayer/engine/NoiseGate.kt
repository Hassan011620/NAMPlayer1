package com.namplayer.engine

import kotlin.math.*

// ─────────────────────────────────────────────────────────────
// Noise Gate
// يقطع الضجيج عند عدم العزف
// Parameters: threshold dB, attack ms, release ms
// ─────────────────────────────────────────────────────────────

class NoiseGate {

    var enabled: Boolean = false

    // Threshold: -70 dB (مغلق تقريباً) → -20 dB (حساس)
    var thresholdDb: Float = -60f
        set(v) { field = v; thresholdLin = 10f.pow(v / 20f) }

    // Attack/Release بـ milliseconds
    var attackMs:  Float = 5f
    var releaseMs: Float = 50f

    private var thresholdLin = 10f.pow(-60f / 20f)
    private var gain         = 0f   // حالة الـ gate: 0=مغلق 1=مفتوح
    private var sampleRate   = 48000

    fun setSampleRate(sr: Int) { sampleRate = sr }

    // ── Process ───────────────────────────────────────────

    fun process(buf: FloatArray, frames: Int) {
        if (!enabled) return

        val attackCoef  = exp(-1f / (attackMs  * sampleRate / 1000f))
        val releaseCoef = exp(-1f / (releaseMs * sampleRate / 1000f))

        for (i in 0 until frames) {
            val level = abs(buf[i])
            gain = if (level > thresholdLin) {
                // Open: رفع الـ gain بسرعة
                1f - attackCoef * (1f - gain)
            } else {
                // Close: إغلاق تدريجي
                releaseCoef * gain
            }
            buf[i] *= gain
        }
    }
}
