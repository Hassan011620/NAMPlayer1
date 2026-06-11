package com.namplayer.usb

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

private const val TAG    = "UsbAudio"
private const val ACTION = "com.namplayer.USB_PERM"

// ─────────────────────────────────────────────
// USB Audio Device info
// ─────────────────────────────────────────────

data class UsbAudioDev(
    val device:   UsbDevice,
    val label:    String,
    val isUac2:   Boolean,
    val maxSr:    Int
) {
    val badge: String  get() = if (isUac2) "UAC2" else "UAC1"
    val icon:  String  get() = if (isUac2) "🟢" else "🟡"
    val info:  String  get() = "$icon $label · ${maxSr/1000}kHz · $badge"
}

// ─────────────────────────────────────────────
// UsbAudioManager
// ─────────────────────────────────────────────

class UsbAudioManager(private val ctx: Context) {

    private val usbMgr = ctx.getSystemService(Context.USB_SERVICE) as UsbManager

    private val _devices = MutableStateFlow<List<UsbAudioDev>>(emptyList())
    val devices: StateFlow<List<UsbAudioDev>> = _devices

    private val _active = MutableStateFlow<UsbAudioDev?>(null)
    val active: StateFlow<UsbAudioDev?> = _active

    fun scan() {
        val found = usbMgr.deviceList.values.mapNotNull { parse(it) }
        _devices.value = found
        Log.d(TAG, "USB scan: ${found.size} audio device(s)")
    }

    fun request(dev: UsbAudioDev) {
        if (usbMgr.hasPermission(dev.device)) { select(dev); return }
        val pi = PendingIntent.getBroadcast(ctx, 0,
            Intent(ACTION), PendingIntent.FLAG_IMMUTABLE)
        usbMgr.requestPermission(dev.device, pi)
    }

    fun select(dev: UsbAudioDev) {
        _active.value = dev
        Log.i(TAG, "Active USB: ${dev.label}")
    }

    fun disconnect() { _active.value = null }

    // ── Parse USB device ──────────────────────────────────

    private fun parse(d: UsbDevice): UsbAudioDev? {
        var hasAudio = false
        for (i in 0 until d.interfaceCount) {
            if (d.getInterface(i).interfaceClass == 1) { hasAudio = true; break }
        }
        if (!hasAudio && !knownVendor(d.vendorId)) return null

        val isUac2 = d.deviceProtocol == 0x20 || d.vendorId == 0x1235
        val label  = d.productName ?: brandName(d.vendorId) ?: "USB Audio"

        return UsbAudioDev(d, label, isUac2, if (isUac2) 96000 else 48000)
    }

    private fun knownVendor(v: Int) = v in setOf(
        0x1235, 0x1397, 0x194F, 0x07FD, 0x2B03,
        0x0E41, 0x0499, 0x2A39, 0x17CC, 0x0763, 0x0582, 0x1F98
    )

    private fun brandName(v: Int) = mapOf(
        0x1235 to "Focusrite", 0x1397 to "Behringer",
        0x194F to "PreSonus",  0x07FD to "MOTU",
        0x2B03 to "IK Multimedia", 0x0E41 to "Line 6",
        0x0499 to "Steinberg", 0x2A39 to "RME",
        0x17CC to "Native Instruments", 0x0763 to "M-Audio",
        0x0582 to "Roland/Boss", 0x1F98 to "Universal Audio"
    )[v]
}

// ─────────────────────────────────────────────
// USB BroadcastReceiver
// ─────────────────────────────────────────────

class UsbReceiver : BroadcastReceiver() {
    override fun onReceive(ctx: Context, intent: Intent) {
        val action = when (intent.action) {
            UsbManager.ACTION_USB_DEVICE_ATTACHED  -> "ATTACHED"
            UsbManager.ACTION_USB_DEVICE_DETACHED  -> "DETACHED"
            ACTION                                 -> "PERMISSION"
            else -> return
        }
        ctx.sendBroadcast(Intent("com.namplayer.USB_$action"))
    }
}
