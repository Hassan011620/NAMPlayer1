package com.namplayer.engine

import com.namplayer.converter.NamModel
import java.nio.ByteBuffer
import java.nio.ByteOrder

// ─────────────────────────────────────────────────────────────
// OnnxBuilder
// يبني ONNX model bytes من NamModel بدون ملف خارجي
// يستخدم ONNX protobuf format مباشرة
// ─────────────────────────────────────────────────────────────

object OnnxBuilder {

    fun build(model: NamModel): ByteArray = when (model) {
        is NamModel.Lstm    -> buildLstmOnnx(model)
        is NamModel.WaveNet -> buildWaveNetOnnx(model)
        is NamModel.Linear  -> buildLinearOnnx(model)
    }

    // ── LSTM ONNX ─────────────────────────────────────────────
    // بنية: input[1,T,1] + h0[L,1,H] + c0[L,1,H]
    //       → output[1,T,1] + hn[L,1,H] + cn[L,1,H]

    private fun buildLstmOnnx(model: NamModel.Lstm): ByteArray {
        val builder = OnnxProtoBuilder()
        builder.setIrVersion(7)
        builder.setOpsetVersion(17)

        val H = model.hiddenSize
        val L = model.numLayers

        // Graph inputs
        builder.addInput("input", OnnxProtoBuilder.FLOAT, listOf(-1, -1, 1))
        builder.addInput("h0",    OnnxProtoBuilder.FLOAT, listOf(L, 1, H))
        builder.addInput("c0",    OnnxProtoBuilder.FLOAT, listOf(L, 1, H))

        // Graph outputs
        builder.addOutput("output", OnnxProtoBuilder.FLOAT, listOf(-1, -1, 1))
        builder.addOutput("hn",     OnnxProtoBuilder.FLOAT, listOf(L, 1, H))
        builder.addOutput("cn",     OnnxProtoBuilder.FLOAT, listOf(L, 1, H))

        // Initializers (weights) per layer
        for (l in 0 until L) {
            val inSz = if (l == 0) 1 else H
            // ONNX LSTM expects W[1, 4H, input_size], R[1, 4H, H], B[1, 8H]
            val wData = padOrSlice(model.wi[l], 4 * H * inSz)
            val rData = padOrSlice(model.wh[l], 4 * H * H)
            val bData = mergeAndPad(model.bi[l], model.bh[l], 8 * H)

            builder.addInitializer("W_$l", floatArrayOf(*wData),
                listOf(1, 4 * H, inSz))
            builder.addInitializer("R_$l", floatArrayOf(*rData),
                listOf(1, 4 * H, H))
            builder.addInitializer("B_$l", floatArrayOf(*bData),
                listOf(1, 8 * H))
        }

        // Head weights
        builder.addInitializer("head_w", model.headW, listOf(1, H))
        builder.addInitializer("head_b", floatArrayOf(model.headB), listOf(1))

        // Nodes: chained LSTM layers + matmul head
        var prevOut = "input"
        var prevH   = "h0"
        var prevC   = "c0"

        for (l in 0 until L) {
            val outY  = if (l == L - 1) "lstm_out" else "lstm_y_$l"
            val outH  = if (l == L - 1) "hn"       else "lstm_h_$l"
            val outC  = if (l == L - 1) "cn"       else "lstm_c_$l"
            val inSz  = if (l == 0) 1 else H

            builder.addNode(
                opType  = "LSTM",
                inputs  = listOf(prevOut, "W_$l", "R_$l", "B_$l", "", prevH, prevC),
                outputs = listOf(outY, outH, outC),
                attrs   = mapOf(
                    "hidden_size" to H,
                    "direction"   to "forward"
                ),
                name = "lstm_$l"
            )
            prevOut = outY; prevH = outH; prevC = outC
        }

        // Squeeze: [1,T,1,H] → [1,T,H]
        builder.addNode("Squeeze", listOf("lstm_out"), listOf("lstm_squeezed"),
            mapOf("axes" to intArrayOf(2)), "squeeze")

        // MatMul: [1,T,H] x [H,1] → [1,T,1]
        builder.addInitializer("head_w_t", model.headW, listOf(H, 1))
        builder.addNode("MatMul", listOf("lstm_squeezed", "head_w_t"), listOf("matmul_out"),
            emptyMap(), "matmul")

        // Add bias
        builder.addNode("Add", listOf("matmul_out", "head_b"), listOf("output"),
            emptyMap(), "add_bias")

        return builder.build()
    }

    // ── WaveNet ONNX ──────────────────────────────────────────
    // WaveNet يُشغَّل native في OnnxInferenceEngine لأن
    // الـ causal dilated conv مع state أصعب في ONNX graph
    // نبني placeholder بسيط يُعيد المدخل (الـ engine يستخدم native)

    private fun buildWaveNetOnnx(model: NamModel.WaveNet): ByteArray {
        val builder = OnnxProtoBuilder()
        builder.setIrVersion(7)
        builder.setOpsetVersion(17)
        builder.addInput("input",  OnnxProtoBuilder.FLOAT, listOf(-1))
        builder.addOutput("output", OnnxProtoBuilder.FLOAT, listOf(-1))
        builder.addInitializer("scale", floatArrayOf(1f), listOf(1))
        builder.addNode("Mul", listOf("input", "scale"), listOf("output"), emptyMap(), "identity")
        return builder.build()
    }

    // ── Linear ONNX ───────────────────────────────────────────

    private fun buildLinearOnnx(model: NamModel.Linear): ByteArray {
        val builder = OnnxProtoBuilder()
        builder.setIrVersion(7)
        builder.setOpsetVersion(17)
        builder.addInput("input",  OnnxProtoBuilder.FLOAT, listOf(-1))
        builder.addOutput("output", OnnxProtoBuilder.FLOAT, listOf(-1))
        builder.addInitializer("w", floatArrayOf(model.w), listOf(1))
        builder.addInitializer("b", floatArrayOf(model.b), listOf(1))
        builder.addNode("Mul", listOf("input", "w"), listOf("mul_out"), emptyMap(), "mul")
        builder.addNode("Add", listOf("mul_out", "b"), listOf("output"), emptyMap(), "add")
        return builder.build()
    }

    // ── Helpers ───────────────────────────────────────────────

    private fun padOrSlice(src: FloatArray, size: Int): FloatArray {
        if (src.size >= size) return src.copyOfRange(0, size)
        return src + FloatArray(size - src.size)
    }

    private fun mergeAndPad(a: FloatArray, b: FloatArray, size: Int): FloatArray {
        val merged = a + b
        return padOrSlice(merged, size)
    }
}

// ─────────────────────────────────────────────────────────────
// Minimal ONNX Protobuf Builder
// يبني ONNX ModelProto bytes بدون مكتبة protobuf خارجية
// ─────────────────────────────────────────────────────────────

class OnnxProtoBuilder {

    companion object {
        const val FLOAT = 1  // TensorProto.DataType.FLOAT
    }

    private val buf = ByteBuffer.allocate(32 * 1024 * 1024) // 32MB
        .order(ByteOrder.LITTLE_ENDIAN)

    // track sections to write at end
    private val inputs        = mutableListOf<ByteArray>()
    private val outputs       = mutableListOf<ByteArray>()
    private val initializers  = mutableListOf<ByteArray>()
    private val nodes         = mutableListOf<ByteArray>()
    private var irVersion     = 7L
    private var opsetVersion  = 17L

    fun setIrVersion(v: Long)    { irVersion = v }
    fun setIrVersion(v: Int)     { irVersion = v.toLong() }
    fun setOpsetVersion(v: Long) { opsetVersion = v }
    fun setOpsetVersion(v: Int)  { opsetVersion = v.toLong() }

    fun addInput(name: String, dtype: Int, shape: List<Int>) {
        inputs.add(makeValueInfo(name, dtype, shape))
    }

    fun addOutput(name: String, dtype: Int, shape: List<Int>) {
        outputs.add(makeValueInfo(name, dtype, shape))
    }

    fun addInitializer(name: String, data: FloatArray, shape: List<Int>) {
        initializers.add(makeTensor(name, data, shape))
    }

    fun addNode(
        opType: String, inputs: List<String>, outputs: List<String>,
        attrs: Map<String, Any>, name: String
    ) {
        nodes.add(makeNode(opType, inputs, outputs, attrs, name))
    }

    fun build(): ByteArray {
        // ModelProto:
        // field 1 = ir_version (int64)
        // field 8 = opset_import (OperatorSetIdProto)
        // field 7 = graph (GraphProto)

        val graphBytes = buildGraph()
        val modelBuf = DynBuf()

        // ir_version: field 1, wire=0 (varint)
        modelBuf.writeTag(1, 0); modelBuf.writeVarint(irVersion)
        // opset_import: field 8, wire=2 (len-delim)
        val opsetBytes = buildOpset()
        modelBuf.writeTag(8, 2); modelBuf.writeBytes(opsetBytes)
        // graph: field 7, wire=2
        modelBuf.writeTag(7, 2); modelBuf.writeBytes(graphBytes)

        return modelBuf.toByteArray()
    }

    private fun buildOpset(): ByteArray {
        // OperatorSetIdProto: field 2 = version (int64)
        val d = DynBuf()
        d.writeTag(2, 0); d.writeVarint(opsetVersion)
        return d.toByteArray()
    }

    private fun buildGraph(): ByteArray {
        // GraphProto:
        // field 1 = node (NodeProto)
        // field 11 = name
        // field 12 = initializer (TensorProto)
        // field 11 = input (ValueInfoProto)  ← field 11 used for both in real proto
        // Actually: node=1, name=2, initializer=5, input=11, output=12
        // Real ONNX: node=1, initializer=5, input=11, output=12, name=2
        val d = DynBuf()
        d.writeTag(2, 2); d.writeBytes("nam_graph".toByteArray())
        for (n in nodes)        { d.writeTag(1,  2); d.writeBytes(n) }
        for (init in initializers) { d.writeTag(5, 2); d.writeBytes(init) }
        for (inp in inputs)     { d.writeTag(11, 2); d.writeBytes(inp) }
        for (out in outputs)    { d.writeTag(12, 2); d.writeBytes(out) }
        return d.toByteArray()
    }

    // ── Proto makers ──────────────────────────────────────────

    private fun makeValueInfo(name: String, dtype: Int, shape: List<Int>): ByteArray {
        // ValueInfoProto: name=1, type=2(TypeProto)
        // TypeProto: tensor_type=1(TypeProto.Tensor)
        // TypeProto.Tensor: elem_type=1, shape=2(TensorShapeProto)
        // TensorShapeProto: dim=1(Dimension)
        // Dimension: dim_value=1 or dim_param=2
        val shapeProto = DynBuf()
        for (d in shape) {
            val dim = DynBuf()
            if (d < 0) { dim.writeTag(2, 2); dim.writeBytes("d".toByteArray()) }
            else       { dim.writeTag(1, 0); dim.writeVarint(d.toLong()) }
            shapeProto.writeTag(1, 2); shapeProto.writeBytes(dim.toByteArray())
        }
        val tensorType = DynBuf()
        tensorType.writeTag(1, 0); tensorType.writeVarint(dtype.toLong())
        tensorType.writeTag(2, 2); tensorType.writeBytes(shapeProto.toByteArray())

        val typeProto = DynBuf()
        typeProto.writeTag(1, 2); typeProto.writeBytes(tensorType.toByteArray())

        val vi = DynBuf()
        vi.writeTag(1, 2); vi.writeBytes(name.toByteArray())
        vi.writeTag(2, 2); vi.writeBytes(typeProto.toByteArray())
        return vi.toByteArray()
    }

    private fun makeTensor(name: String, data: FloatArray, shape: List<Int>): ByteArray {
        // TensorProto: dims=1, data_type=2, float_data=4, name=8
        val t = DynBuf()
        for (d in shape) { t.writeTag(1, 0); t.writeVarint(d.toLong()) }
        t.writeTag(2, 0); t.writeVarint(1) // FLOAT
        // raw_data field 9 — more compact
        val raw = ByteBuffer.allocate(data.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        data.forEach { raw.putFloat(it) }
        t.writeTag(9, 2); t.writeBytes(raw.array())
        t.writeTag(8, 2); t.writeBytes(name.toByteArray())
        return t.toByteArray()
    }

    private fun makeNode(
        opType: String, ins: List<String>, outs: List<String>,
        attrs: Map<String, Any>, name: String
    ): ByteArray {
        // NodeProto: input=1, output=2, name=3, op_type=4, attribute=5
        val n = DynBuf()
        for (i in ins)  { n.writeTag(1, 2); n.writeBytes(i.toByteArray()) }
        for (o in outs) { n.writeTag(2, 2); n.writeBytes(o.toByteArray()) }
        n.writeTag(3, 2); n.writeBytes(name.toByteArray())
        n.writeTag(4, 2); n.writeBytes(opType.toByteArray())
        for ((k, v) in attrs) {
            n.writeTag(5, 2); n.writeBytes(makeAttr(k, v))
        }
        return n.toByteArray()
    }

    private fun makeAttr(name: String, value: Any): ByteArray {
        // AttributeProto: name=1, f=4, i=3, ints=7, floats=6, type=20
        val a = DynBuf()
        a.writeTag(1, 2); a.writeBytes(name.toByteArray())
        when (value) {
            is Int    -> { a.writeTag(20, 0); a.writeVarint(1); a.writeTag(3, 0); a.writeVarint(value.toLong()) }
            is Long   -> { a.writeTag(20, 0); a.writeVarint(1); a.writeTag(3, 0); a.writeVarint(value) }
            is Float  -> { a.writeTag(20, 0); a.writeVarint(4); a.writeTag(4, 5); a.writeFixed32(value.toBits()) }
            is String -> { a.writeTag(20, 0); a.writeVarint(8); a.writeTag(13, 2); a.writeBytes(value.toByteArray()) }
            is IntArray -> {
                a.writeTag(20, 0); a.writeVarint(7)
                for (v in value) { a.writeTag(7, 0); a.writeVarint(v.toLong()) }
            }
            else -> {}
        }
        return a.toByteArray()
    }
}

// ── Dynamic Buffer helper ────────────────────────────────────

class DynBuf {
    private val data = mutableListOf<Byte>()

    fun writeVarint(v: Long) {
        var rem = v
        do {
            var b = (rem and 0x7F).toByte()
            rem = rem ushr 7
            if (rem != 0L) b = (b.toInt() or 0x80).toByte()
            data.add(b)
        } while (rem != 0L)
    }

    fun writeTag(fieldNum: Int, wireType: Int) = writeVarint(((fieldNum shl 3) or wireType).toLong())

    fun writeBytes(bytes: ByteArray) {
        writeVarint(bytes.size.toLong())
        bytes.forEach { data.add(it) }
    }

    fun writeFixed32(v: Int) {
        data.add((v and 0xFF).toByte())
        data.add(((v shr 8) and 0xFF).toByte())
        data.add(((v shr 16) and 0xFF).toByte())
        data.add(((v shr 24) and 0xFF).toByte())
    }

    fun toByteArray(): ByteArray = data.toByteArray()
}
