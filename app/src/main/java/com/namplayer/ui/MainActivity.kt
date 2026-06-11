package com.namplayer.ui

import android.Manifest
import android.content.*
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.namplayer.audio.EngineState
import com.namplayer.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var b: ActivityMainBinding
    private val vm: MainViewModel by viewModels()

    // ── File pickers ──────────────────────────────────────

    private val pickNam = registerForActivityResult(
        ActivityResultContracts.OpenDocument()) { uri: Uri? -> uri?.let { vm.loadNam(it) } }

    private val pickCab = registerForActivityResult(
        ActivityResultContracts.OpenDocument()) { uri: Uri? -> uri?.let { vm.loadCab(it) } }

    private val askPerm = registerForActivityResult(
        ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) vm.toggle() else toast("يجب منح صلاحية الميكروفون")
    }

    // ── USB receiver ──────────────────────────────────────

    private val usbRx = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context, i: Intent) = when (i.action) {
            "com.namplayer.USB_ATTACHED"   -> vm.onUsbAttached()
            "com.namplayer.USB_DETACHED"   -> vm.onUsbDetached()
            "com.namplayer.USB_PERMISSION" -> vm.onUsbAttached()
            else -> Unit
        }
    }

    // ── onCreate ──────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)

        wire()
        observe()
        registerReceiver(usbRx, IntentFilter().apply {
            addAction("com.namplayer.USB_ATTACHED")
            addAction("com.namplayer.USB_DETACHED")
            addAction("com.namplayer.USB_PERMISSION")
        }, RECEIVER_NOT_EXPORTED)

        intent?.data?.let { vm.loadNam(it) }
    }

    override fun onDestroy() { super.onDestroy(); unregisterReceiver(usbRx) }

    // ── Wire ─────────────────────────────────────────────

    private fun wire() {
        // Play/Stop
        b.btnPlay.setOnClickListener {
            if (!hasMicPerm()) { askPerm.launch(Manifest.permission.RECORD_AUDIO); return@setOnClickListener }
            vm.toggle()
        }

        // Open .nam
        b.btnOpen.setOnClickListener { pickNam.launch(arrayOf("*/*")) }

        // USB
        b.btnUsb.setOnClickListener { showUsbSheet() }

        // Gains
        b.sliderIn.addOnChangeListener  { _, v, u -> if (u) { vm.setInputGain(v);  b.tvIn.text  = "${v.toInt()}dB" } }
        b.sliderOut.addOnChangeListener { _, v, u -> if (u) { vm.setOutputGain(v); b.tvOut.text = "${v.toInt()}dB" } }

        // Cabinet IR
        b.btnCab.setOnClickListener { showCabSheet() }

        // Noise Gate toggle
        b.switchGate.setOnCheckedChangeListener { _, on ->
            vm.setGateEnabled(on)
            b.sliderGate.isEnabled = on
        }
        b.sliderGate.addOnChangeListener { _, v, u ->
            if (u) { vm.setGateThreshold(v); b.tvGate.text = "${v.toInt()}dB" }
        }

        // Record button
        b.btnRecord.setOnClickListener { vm.toggleRecord() }
    }

    // ── Observe ───────────────────────────────────────────

    private fun observe() {
        lifecycleScope.launch {
            vm.engineState.collect { state ->
                when (state) {
                    EngineState.RUNNING -> {
                        b.btnPlay.text = "⏹"
                        b.btnPlay.setBackgroundColor(0xFFE53935.toInt())
                        b.dot.setBackgroundResource(com.namplayer.R.drawable.dot_green)
                        b.tvState.text = "يعمل"
                    }
                    EngineState.IDLE -> {
                        b.btnPlay.text = "▶"
                        b.btnPlay.setBackgroundColor(0xFF43A047.toInt())
                        b.dot.setBackgroundResource(com.namplayer.R.drawable.dot_grey)
                        b.tvState.text = "متوقف"
                    }
                    EngineState.ERROR -> {
                        b.dot.setBackgroundResource(com.namplayer.R.drawable.dot_red)
                        b.tvState.text = "خطأ"
                        toast("خطأ في الصوت")
                    }
                }
            }
        }

        lifecycleScope.launch {
            vm.model.collect { mdl ->
                b.tvModel.text = mdl?.displayName ?: "لا يوجد موديل"
                b.tvArch.text  = mdl?.arch ?: ""
            }
        }

        lifecycleScope.launch {
            vm.cabName.collect { name ->
                b.tvCab.text = name ?: "Cabinet: off"
                b.tvCab.setTextColor(if (name != null) 0xFF43A047.toInt() else 0xFF666666.toInt())
            }
        }

        lifecycleScope.launch {
            vm.stats.collect { s ->
                b.tvLatency.text = "%.1fms".format(s.latencyMs)
                b.tvCpu.text     = "%.0f%%".format(s.cpuPct)
                b.tvSr.text      = "${s.sampleRate/1000}kHz"

                b.vuIn.progress  = ((s.inLevelDb  + 60f) / 60f * 100f).toInt().coerceIn(0, 100)
                b.vuOut.progress = ((s.outLevelDb + 60f) / 60f * 100f).toInt().coerceIn(0, 100)

                b.tvXrun.visibility = if (s.xruns > 0) View.VISIBLE else View.GONE
                b.tvXrun.text = "⚠ xrun:${s.xruns}"

                // Record button color + timer
                if (s.isRecording) {
                    b.btnRecord.text = "⏹ ${s.recSeconds}s"
                    b.btnRecord.setBackgroundColor(0xFFE53935.toInt())
                } else {
                    b.btnRecord.text = "⏺ REC"
                    b.btnRecord.setBackgroundColor(0xFF333333.toInt())
                }
            }
        }

        lifecycleScope.launch {
            vm.activeUsb.collect { d ->
                b.btnUsb.text = if (d != null) "🔌 ${d.label}" else "🔌 USB"
            }
        }

        lifecycleScope.launch {
            vm.loading.collect { b.progress.visibility = if (it) View.VISIBLE else View.GONE }
        }

        lifecycleScope.launch { vm.msg.collect { toast(it) } }
    }

    // ── Cabinet Sheet ─────────────────────────────────────

    private fun showCabSheet() {
        val sheet = com.google.android.material.bottomsheet.BottomSheetDialog(this)
        val v = layoutInflater.inflate(com.namplayer.R.layout.sheet_cab, null)
        sheet.setContentView(v)

        val cabName = vm.cabName.value
        v.findViewById<TextView>(com.namplayer.R.id.tvCabCurrent).text =
            cabName ?: "لا يوجد Cabinet محمّل"

        v.findViewById<Button>(com.namplayer.R.id.btnLoadCab).setOnClickListener {
            pickCab.launch(arrayOf("audio/*", "*/*"))
            sheet.dismiss()
        }
        v.findViewById<Button>(com.namplayer.R.id.btnClearCab).setOnClickListener {
            vm.clearCab(); sheet.dismiss()
        }
        // Enable/disable toggle
        val sw = v.findViewById<Switch>(com.namplayer.R.id.switchCab)
        sw.isChecked = vm.cabName.value != null
        sw.setOnCheckedChangeListener { _, on -> vm.setCabEnabled(on) }

        sheet.show()
    }

    // ── USB Sheet ─────────────────────────────────────────

    private fun showUsbSheet() {
        val sheet = com.google.android.material.bottomsheet.BottomSheetDialog(this)
        val v = layoutInflater.inflate(com.namplayer.R.layout.sheet_usb, null)
        sheet.setContentView(v)

        val list  = v.findViewById<LinearLayout>(com.namplayer.R.id.llDevices)
        val empty = v.findViewById<TextView>(com.namplayer.R.id.tvEmpty)
        val devs  = vm.usbDevices.value

        if (devs.isEmpty()) { empty.visibility = View.VISIBLE }
        else {
            empty.visibility = View.GONE
            devs.forEach { dev ->
                val row = layoutInflater.inflate(com.namplayer.R.layout.item_usb, list, false)
                row.findViewById<TextView>(com.namplayer.R.id.tvDevName).text = dev.info
                row.setOnClickListener { vm.selectUsb(dev); sheet.dismiss() }
                list.addView(row)
            }
        }
        v.findViewById<Button>(com.namplayer.R.id.btnScan).setOnClickListener {
            vm.onUsbAttached(); sheet.dismiss(); showUsbSheet()
        }
        sheet.show()
    }

    // ── Helpers ───────────────────────────────────────────

    private fun hasMicPerm() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
}
