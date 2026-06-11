package com.namplayer.converter

import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

private const val TAG = "NamConverter"

// ─────────────────────────────────────────────
// .nam JSON structures
// ─────────────────────────────────────────────

data class NamJson(
    @SerializedName("version")      val version: String = "0.0.0",
    @SerializedName("architecture") val architecture: String,
    @SerializedName("config")       val config: Map<String, Any>,
    @SerializedName("weights")      val weights: List<Double>,
    @SerializedName("sample_rate")  val sampleRate: Int? = 48000,
    @SerializedName("metadata")     val metadata: NamMeta? = null
)

data class NamMeta(
    @SerializedName("name")        val name: String? = null,
    @SerializedName("loudness")    val loudness: Double? = null,
    @SerializedName("gain")        val gain: Double? = null,
    @SerializedName("gear_make")   val gearMake: String? = null,
    @SerializedName("gear_model")  val gearModel: String? = null,
    @SerializedName("gear_type")   val gearType: String? = null,
    @SerializedName("tone_type")   val toneType: String? = null,
    @SerializedName("modeled_by")  val modeledBy: String? = null
)

// ─────────────────────────────────────────────
// Parsed model ready for inference
// ─────────────────────────────────────────────

sealed class NamModel {
    abstract val sampleRate: Int
    abstract val loudnessDb: Float
    abstract val displayName: String
    abstract val arch: String

    data class Lstm(
        val hiddenSize: Int,
        val numLayers: Int,
        // weights: Wi[layer], Wh[layer], bi[layer], bh[layer]
        val wi: Array<FloatArray>,
        val wh: Array<FloatArray>,
        val bi: Array<FloatArray>,
        val bh: Array<FloatArray>,
        val headW: FloatArray,
        val headB: Float,
        override val sampleRate: Int,
        override val loudnessDb: Float,
        override val displayName: String,
        override val arch: String = "LSTM"
    ) : NamModel()

    data class WaveNet(
        val channels: Int,
        val kernelSize: Int,
        val dilations: List<Int>,
        // Flattened conv weights per dilation layer
        val convWeights: Array<FloatArray>,
        val convBiases: Array<FloatArray>,
        val headW: FloatArray,
        val headB: Float,
        override val sampleRate: Int,
        override val loudnessDb: Float,
        override val displayName: String,
        override val arch: String = "WaveNet"
    ) : NamModel()

    data class Linear(
        val w: Float,
        val b: Float,
        override val sampleRate: Int,
        override val loudnessDb: Float,
        override val displayName: String,
        override val arch: String = "Linear"
    ) : NamModel()
}

// ─────────────────────────────────────────────
// Parser — .nam → NamModel
// ─────────────────────────────────────────────

object NamParser {
    private val gson = Gson()

    fun parse(stream: InputStream): Result<NamJson> = runCatching {
        gson.fromJson(stream.bufferedReader(), NamJson::class.java)
            ?: throw IllegalStateException("Empty .nam file")
    }

    fun build(json: NamJson): Result<NamModel> = runCatching {
        val w = json.weights.map { it.toFloat() }.toFloatArray()
        val sr = json.sampleRate ?: 48000
        val loudness = json.metadata?.loudness?.toFloat() ?: -18f
        val displayName = buildDisplayName(json)

        when (json.architecture.uppercase()) {
            "LSTM"    -> buildLstm(json.config, w, sr, loudness, displayName)
            "WAVENET" -> buildWaveNet(json.config, w, sr, loudness, displayName)
            "LINEAR"  -> NamModel.Linear(
                w = if (w.isNotEmpty()) w[0] else 1f,
                b = if (w.size > 1) w[1] else 0f,
                sampleRate = sr, loudnessDb = loudness, displayName = displayName
            )
            else -> throw IllegalArgumentException("Unknown arch: ${json.architecture}")
        }
    }

    // ── LSTM builder ────────────────────────────────────────

    @Suppress("UNCHECKED_CAST")
    private fun buildLstm(
        config: Map<String, Any>, w: FloatArray,
        sr: Int, loudness: Float, name: String
    ): NamModel.Lstm {
        val hidden = (config["hidden_size"] as? Double)?.toInt() ?: 16
        val layers = (config["num_layers"]  as? Double)?.toInt() ?: 1
        val gateSize = 4 * hidden

        val wi = Array(layers) { FloatArray(0) }
        val wh = Array(layers) { FloatArray(0) }
        val bi = Array(layers) { FloatArray(0) }
        val bh = Array(layers) { FloatArray(0) }

        var idx = 0
        fun slice(n: Int): FloatArray {
            val arr = w.copyOfRange(idx.coerceAtMost(w.size), (idx + n).coerceAtMost(w.size))
            idx += n
            return arr.let { if (it.size < n) it + FloatArray(n - it.size) else it }
        }

        for (l in 0 until layers) {
            val inSz = if (l == 0) 1 else hidden
            wi[l] = slice(gateSize * inSz)
            wh[l] = slice(gateSize * hidden)
            bi[l] = slice(gateSize)
            bh[l] = slice(gateSize)
        }
        val headW = slice(hidden)
        val headB = if (idx < w.size) w[idx] else 0f

        return NamModel.Lstm(hidden, layers, wi, wh, bi, bh, headW, headB, sr, loudness, name)
    }

    // ── WaveNet builder ─────────────────────────────────────

    @Suppress("UNCHECKED_CAST")
    private fun buildWaveNet(
        config: Map<String, Any>, w: FloatArray,
        sr: Int, loudness: Float, name: String
    ): NamModel.WaveNet {
        val channels   = (config["channels"]    as? Double)?.toInt() ?: 8
        val kernelSize = (config["kernel_size"] as? Double)?.toInt() ?: 3
        val rawDil     = config["dilations"] as? List<*>
        val dilations  = rawDil?.map { (it as Double).toInt() } ?: listOf(1, 2, 4, 8)

        val convWeights = Array(dilations.size) { FloatArray(0) }
        val convBiases  = Array(dilations.size) { FloatArray(0) }

        var idx = 0
        fun slice(n: Int): FloatArray {
            val end = (idx + n).coerceAtMost(w.size)
            val arr = w.copyOfRange(idx, end)
            idx += n
            return if (arr.size < n) arr + FloatArray(n - arr.size) else arr
        }

        for (i in dilations.indices) {
            val inCh = if (i == 0) 1 else channels
            convWeights[i] = slice(channels * inCh * kernelSize)
            convBiases[i]  = slice(channels)
        }
        // head 1x1
        val headW = slice(channels)
        val headB = if (idx < w.size) w[idx] else 0f

        return NamModel.WaveNet(channels, kernelSize, dilations,
            convWeights, convBiases, headW, headB, sr, loudness, name)
    }

    // ── Display name ────────────────────────────────────────

    private fun buildDisplayName(json: NamJson): String {
        val m = json.metadata ?: return json.architecture
        return m.name ?: buildString {
            m.gearMake?.let  { append(it).append(" ") }
            m.gearModel?.let { append(it) }
            if (isEmpty()) append(json.architecture)
        }
    }
}
