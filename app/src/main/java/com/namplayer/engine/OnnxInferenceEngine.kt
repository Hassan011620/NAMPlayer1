package com.namplayer.engine

import ai.onnxruntime.*
import android.content.Context
import android.util.Log
import com.namplayer.converter.NamModel
import java.nio.FloatBuffer
import kotlin.math.*

private const val TAG = "OnnxEngine"

// ─────────────────────────────────────────────────────────────
// ONNX Inference Engine
// يحوّل NamModel → ONNX session ويشغّل الـ inference
// ─────────────────────────────────────────────────────────────

class OnnxInferenceEngine(context: Context) {

    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var currentModel: NamModel? = null

    // LSTM state (يُعاد استخدامه بين الـ buffers)
    private var lstmH: Array<FloatArray> = emptyArray()
    private var lstmC: Array<FloatArray> = emptyArray()

    // WaveNet dilated buffers
    private var waveBuffers: Array<FloatArray> = emptyArray()
    private var waveBufPos:  IntArray = intArrayOf()

    // Gain
    private var inputGain:  Float = 1f
    private var outputGain: Float = 1f

    val isLoaded: Boolean get() = currentModel != null

    // ── Load model ────────────────────────────────────────────

    fun load(model: NamModel): Boolean {
        unload()
        return runCatching {
            // بناء ONNX model bytes من الـ NamModel
            val onnxBytes = OnnxBuilder.build(model)
            val opts = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(2)
                // NNAPI acceleration إذا كان متاحاً
                try { addNnapi() } catch (_: Exception) {}
            }
            session = ortEnv.createSession(onnxBytes, opts)
            currentModel = model

            // loudness normalization → target -18 dBFS
            val loudness = model.loudnessDb
            outputGain = 10f.pow((-18f - loudness) / 20f).coerceIn(0.1f, 8f)

            // تهيئة state buffers
            resetState(model)

            Log.i(TAG, "Loaded ${model.arch} '${model.displayName}' " +
                  "SR=${model.sampleRate} loudness=${model.loudnessDb}")
            true
        }.onFailure { Log.e(TAG, "Load failed", it) }
         .getOrDefault(false)
    }

    fun unload() {
        session?.close(); session = null
        currentModel = null
        lstmH = emptyArray(); lstmC = emptyArray()
        waveBuffers = emptyArray()
    }

    // ── Process audio buffer ──────────────────────────────────

    fun process(input: FloatArray, output: FloatArray, frames: Int) {
        val mdl = currentModel ?: run { output.fill(0f, 0, frames); return }
        val sess = session

        // Fallback ke native DSP jika session null
        if (sess == null) {
            processFallback(mdl, input, output, frames)
            return
        }

        runCatching {
            when (mdl) {
                is NamModel.Lstm    -> processLstm(sess, mdl, input, output, frames)
                is NamModel.WaveNet -> processWaveNet(mdl, input, output, frames)
                is NamModel.Linear  -> processLinear(mdl, input, output, frames)
            }
        }.onFailure {
            Log.w(TAG, "ONNX error, fallback: ${it.message}")
            processFallback(mdl, input, output, frames)
        }
    }

    // ── LSTM via ONNX ─────────────────────────────────────────

    private fun processLstm(
        sess: OrtSession, mdl: NamModel.Lstm,
        input: FloatArray, output: FloatArray, frames: Int
    ) {
        // Input tensor: [1, frames, 1]
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(applyInputGain(input, frames)),
            longArrayOf(1, frames.toLong(), 1)
        )
        // h0, c0 tensors: [numLayers, 1, hiddenSize]
        val h0Flat = lstmH.flatMap { it.toList() }.toFloatArray()
        val c0Flat = lstmC.flatMap { it.toList() }.toFloatArray()
        val h0 = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(h0Flat),
            longArrayOf(mdl.numLayers.toLong(), 1, mdl.hiddenSize.toLong()))
        val c0 = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(c0Flat),
            longArrayOf(mdl.numLayers.toLong(), 1, mdl.hiddenSize.toLong()))

        val inputs = mapOf("input" to inputTensor, "h0" to h0, "c0" to c0)
        val results = sess.run(inputs)

        // Output: [1, frames, 1]
        val outTensor = results[0].value as Array<*>
        var outIdx = 0
        for (t in 0 until frames) {
            val v = ((outTensor[0] as Array<*>)[t] as FloatArray)[0]
            output[t] = v * outputGain
            outIdx++
        }

        // Update LSTM state from results
        updateLstmState(results, mdl)

        inputTensor.close(); h0.close(); c0.close(); results.close()
    }

    private fun updateLstmState(results: OrtSession.Result, mdl: NamModel.Lstm) {
        runCatching {
            val hn = results[1].value as Array<*>
            val cn = results[2].value as Array<*>
            for (l in 0 until mdl.numLayers) {
                val hLayer = (hn[l] as Array<*>)[0] as FloatArray
                val cLayer = (cn[l] as Array<*>)[0] as FloatArray
                hLayer.copyInto(lstmH[l])
                cLayer.copyInto(lstmC[l])
            }
        }
    }

    // ── WaveNet Native (أسرع بدون ONNX overhead للـ causal conv) ──

    private fun processWaveNet(
        mdl: NamModel.WaveNet,
        input: FloatArray, output: FloatArray, frames: Int
    ) {
        val ch = mdl.channels
        val ks = mdl.kernelSize

        for (t in 0 until frames) {
            var x = input[t] * inputGain
            val intermediate = FloatArray(ch)

            mdl.dilations.forEachIndexed { li, dilation ->
                val bufSize = ks * dilation
                val buf = waveBuffers[li]
                val pos = waveBufPos[li]
                val inCh = if (li == 0) 1 else ch

                // push x into buffer
                if (inCh == 1) {
                    buf[pos % bufSize] = x
                } else {
                    for (c in 0 until ch) buf[(pos % bufSize) * ch + c] = intermediate[c]
                }

                // dilated conv
                val cw = mdl.convWeights[li]
                val cb = mdl.convBiases[li]
                for (oc in 0 until ch) {
                    var sum = cb[oc]
                    for (k in 0 until ks) {
                        val si = ((pos - k * dilation + bufSize * ks) % bufSize)
                        if (inCh == 1) {
                            sum += cw[oc * ks + k] * buf[si]
                        } else {
                            for (ic in 0 until ch) {
                                sum += cw[oc * inCh * ks + ic * ks + k] * buf[si * ch + ic]
                            }
                        }
                    }
                    intermediate[oc] = relu(sum)
                }
                waveBufPos[li] = (pos + 1) % bufSize
            }

            // head 1x1
            var y = mdl.headB
            for (c in 0 until ch) y += mdl.headW[c] * intermediate[c]
            output[t] = y * outputGain
        }
    }

    // ── Linear ────────────────────────────────────────────────

    private fun processLinear(mdl: NamModel.Linear, input: FloatArray, output: FloatArray, frames: Int) {
        for (i in 0 until frames)
            output[i] = (mdl.w * input[i] * inputGain + mdl.b) * outputGain
    }

    // ── Fallback Native DSP (بدون ONNX) ──────────────────────

    private fun processFallback(mdl: NamModel, input: FloatArray, output: FloatArray, frames: Int) {
        when (mdl) {
            is NamModel.Lstm    -> processLstmNative(mdl, input, output, frames)
            is NamModel.WaveNet -> processWaveNet(mdl, input, output, frames)
            is NamModel.Linear  -> processLinear(mdl, input, output, frames)
        }
    }

    // ── Native LSTM (fallback) ────────────────────────────────

    private fun processLstmNative(
        mdl: NamModel.Lstm, input: FloatArray, output: FloatArray, frames: Int
    ) {
        val hidden = mdl.hiddenSize
        val gateSize = 4 * hidden
        val gates = FloatArray(gateSize)
        val newH  = FloatArray(hidden)

        for (t in 0 until frames) {
            var x = input[t] * inputGain
            for (l in 0 until mdl.numLayers) {
                val h = lstmH[l]; val c = lstmC[l]
                val wi = mdl.wi[l]; val wh = mdl.wh[l]
                val bi = mdl.bi[l]; val bh = mdl.bh[l]
                val inSz = if (l == 0) 1 else hidden

                // gates
                for (g in 0 until gateSize) {
                    var s = bi[g] + bh[g]
                    if (inSz == 1) s += wi[g] * x
                    else for (j in 0 until hidden) s += wi[g * hidden + j] * (if (l == 0) 0f else lstmH[l-1][j])
                    for (j in 0 until hidden) s += wh[g * hidden + j] * h[j]
                    gates[g] = s
                }
                for (j in 0 until hidden) {
                    val ig = sigmoid(gates[j])
                    val fg = sigmoid(gates[hidden + j])
                    val gg = tanh(gates[2 * hidden + j].toDouble()).toFloat()
                    val og = sigmoid(gates[3 * hidden + j])
                    c[j] = fg * c[j] + ig * gg
                    newH[j] = og * tanh(c[j].toDouble()).toFloat()
                }
                newH.copyInto(h)
            }
            var y = mdl.headB
            for (j in 0 until hidden) y += mdl.headW[j] * lstmH[mdl.numLayers - 1][j]
            output[t] = y * outputGain
        }
    }

    // ── Helpers ───────────────────────────────────────────────

    private fun resetState(model: NamModel) {
        when (model) {
            is NamModel.Lstm -> {
                lstmH = Array(model.numLayers) { FloatArray(model.hiddenSize) }
                lstmC = Array(model.numLayers) { FloatArray(model.hiddenSize) }
            }
            is NamModel.WaveNet -> {
                waveBuffers = Array(model.dilations.size) { i ->
                    val inCh = if (i == 0) 1 else model.channels
                    FloatArray(model.kernelSize * model.dilations[i] * inCh)
                }
                waveBufPos = IntArray(model.dilations.size)
            }
            else -> {}
        }
    }

    fun resetState() { currentModel?.let { resetState(it) } }

    private fun applyInputGain(input: FloatArray, frames: Int): FloatArray {
        if (inputGain == 1f) return input.copyOfRange(0, frames)
        return FloatArray(frames) { input[it] * inputGain }
    }

    fun setInputGainDb(db: Float)  { inputGain  = 10f.pow(db / 20f) }
    fun setOutputGainDb(db: Float) { outputGain = 10f.pow(db / 20f) }

    private fun sigmoid(x: Float) = 1f / (1f + exp(-x))
    private fun relu(x: Float)    = if (x > 0f) x else 0f

    fun close() { unload(); ortEnv.close() }
}
