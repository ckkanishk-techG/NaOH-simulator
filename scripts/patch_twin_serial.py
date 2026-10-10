"""Add browser Web Serial (Arduino/ESP32) as a live source for the prototype's Live Twin page.

Patches ``prototype/hydra_l1_prototype.html`` in place (the page is a bundle: gzip+base64 assets plus a JSON-encoded template)
and ``prototype/hydra_l1_model.js`` (the readable copy of the model). Idempotent: refuses to patch twice.

Usage: python scripts/patch_twin_serial.py [prototype/hydra_l1_prototype.html]
"""

from __future__ import annotations

import base64
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ASSET = "2e647804-6aa8-4f5e-a523-cc5cea2353cc"
MARKER = "liveIngest"


def once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"expected exactly one match for: {old[:70]!r} (found {text.count(old)})"
    return text.replace(old, new)


# ------------------------------------------------------------------------------------------------ model (JS module)
MODEL_NEW_FUNCS = r"""// ---------- live serial (Web Serial) ----------
// One CSV line per sample, same format as the Arduino logger: t_s,T1_C,T2_C,P_bar_g,flow_L_min,I_A,V_V  (nan / blank = missing).
export function parseFrame(line) {
  const s = line.trim(); if (!s || s[0] === '#' || /^t_(s|ms)/.test(s)) return null;
  const p = s.split(','); if (p.length < 2) return null;
  const v = [];
  for (let i = 0; i < 7; i++) {
    const x = (p[i] === undefined ? '' : p[i]).trim().toLowerCase();
    if (x === '' || x === 'nan' || x === 'na' || x === 'null') { v.push(NaN); continue; }
    const n = Number(x); if (!isFinite(n)) return null; v.push(n);
  }
  if (!isFinite(v[0])) return null;
  return { t: v[0], T1: v[1], T2: v[2], P: v[3], flow: v[4], I: v[5], V: v[6] };
}
export function liveCreate() { return { kind: 'live', latest: null, tRel: 0, tPrev: null, cumL: 0, frames: [], nFrames: 0, lines: 0, gaps: 0, resets: 0, tauT: 15 }; }
// Cumulative H2 [mL STP] = integral of the flow column (L/min at STP), gaps longer than 5 s are NOT integrated (counted in .gaps).
// Time is the logger's own clock (relative), so a stalled browser does not distort it; a logger reboot (t decreasing) is counted in .resets.
export function liveIngest(src, fr) {
  src.nFrames++;
  if (src.tPrev == null) src.tPrev = fr.t;
  const dt = fr.t - src.tPrev; src.tPrev = fr.t;
  if (dt < 0) src.resets++;
  else if (dt > 0) { src.tRel += dt; if (isFinite(fr.flow)) { if (dt > 5) src.gaps++; src.cumL += fr.flow * Math.min(dt, 5) / 60; } } // flow is integrated as reported (no clamping at 0: clamping turns zero-mean noise into a positive volume bias)
  if (src.frames.length < 100000) src.frames.push(fr);
  src.latest = { t: src.tRel, v: src.cumL * 1000, T: fr.T1, fr };
  return src;
}

"""


def patch_model_js(code: str) -> str:
    assert MARKER not in code, "model already patched"
    code = once(code, "// ---------- live twin (EnKF) ----------\n", MODEL_NEW_FUNCS + "// ---------- live twin (EnKF) ----------\n")
    code = once(code, "const rng = mulberry32(seed), lag = src.kind === 'data' ? src.run.tauT : 15;",
                "const rng = mulberry32(seed), lag = src.kind === 'data' ? src.run.tauT : (src.tauT || 15);")
    code = once(code, "hasT: src.kind === 'simtest' || !!(src.run && src.run.T && src.run.sT) };",
                "hasT: src.kind === 'simtest' || src.kind === 'live' || !!(src.run && src.run.T && src.run.sT) };")
    code = once(code, "export function twinTick(tw, adv = 20) {\n",
                "export function twinTick(tw, adv = 20) {\n  if (tw.src.kind === 'live' && !tw.src.latest) return tw; // nothing received yet\n")
    code = once(code, "  else { const r = tw.src.run; sd = [r.sv];",
                "  else if (tw.src.kind === 'live') { const L = tw.src.latest; sd = [Math.max(3, 0.03 * L.v)]; y = [L.v]; if (isFinite(L.T)) { sd.push(0.5); y.push(L.T); } } // assumed sensor noise\n"
                "  else { const r = tw.src.run; sd = [r.sv];")
    return code


# ------------------------------------------------------------------------------------------------ template (page logic + markup)
# One shared serial connection (this._live) used by the Scenario page ("Get from sensors") and the Live Twin page.
LIVE_METHODS = r"""  async closeLive() {
    const l = this._live; this._live = null; clearInterval(this._liveUi);
    if (!l || !l.port) return;
    try { if (l.reader) await l.reader.cancel(); } catch (e) { }
    try { if (l.closedP) await l.closedP; } catch (e) { }
    try { await l.port.close(); } catch (e) { }
  }
  disconnectLive() {
    clearInterval(this._iv); const wasLive = this.tw && this.tw.src.kind === 'live'; if (wasLive) this.tw = null;
    this.closeLive(); this.setState(s => ({ twOn: false, twSrc: wasLive ? null : s.twSrc, twN: s.twN + 1 }));
  }
  async serialConnect() {
    const st = this.state, M = this.M;
    if (!(typeof navigator !== 'undefined' && 'serial' in navigator)) { this.setState({ twErr: 'This browser has no Web Serial. Use desktop Chrome or Edge (page must be https or localhost).' }); return; }
    let port;
    try { port = await navigator.serial.requestPort(); await port.open({ baudRate: st.twBaud || 115200 }); }
    catch (e) { this.setState({ twErr: e && e.name === 'NotFoundError' ? 'No port selected.' : 'Could not open the port (is another program, e.g. the Arduino IDE serial monitor, using it?): ' + (e && e.message || e) }); return; }
    await this.closeLive();
    const live = M.liveCreate(); live.port = port; live.baud = st.twBaud || 115200; live.err = ''; live.lastWall = null;
    const info = port.getInfo ? port.getInfo() : {};
    live.name = info.usbVendorId != null ? 'USB ' + info.usbVendorId.toString(16) + ':' + (info.usbProductId || 0).toString(16) : 'serial port';
    this._live = live;
    (async () => {
      const dec = new TextDecoderStream(); live.closedP = port.readable.pipeTo(dec.writable).catch(() => { });
      live.reader = dec.readable.getReader(); let buf = '';
      try {
        for (; ;) {
          const { value, done } = await live.reader.read(); if (done) break; buf += value;
          let i; while ((i = buf.indexOf('\n')) >= 0) { const ln = buf.slice(0, i); buf = buf.slice(i + 1); if (ln.trim()) live.lines++; const fr = M.parseFrame(ln); if (fr) { M.liveIngest(live, fr); live.lastWall = Date.now(); } }
          if (buf.length > 4096) buf = '';
        }
      } catch (e) { live.err = 'Serial read stopped: ' + (e && e.message || e); }
      if (!live.err && this._live === live) live.err = 'Serial port closed (device unplugged?).';
      live.ended = true;
    })();
    clearInterval(this._liveUi); this._liveUi = setInterval(() => this.setState(x => ({ twN: x.twN + 1 })), 1000);
    this.setState({ twErr: '' });
  }
  startTwinLive() {
    const st = this.state, M = this.M, R = st.runSc, live = this._live;
    if (!R || !live) return;
    clearInterval(this._iv);
    live.cumL = 0; live.tRel = 0; live.latest = null; live.gaps = 0; live.resets = 0; // t = 0 and zero volume from this moment
    const sc = { ...R, mode: 'open', cells: 0, cellArea: 1, load: 'none', pRelief: 0 };
    this.tw = M.twinCreate(this.base(), 24, st.seed, sc, live);
    this._iv = setInterval(() => {
      const tw2 = this.tw, L = tw2 && tw2.src;
      if (L && L.latest) { const adv = Math.min(Math.floor((L.latest.t - tw2.members[0].s.t) / 2) * 2, 120); if (adv >= 2) M.twinTick(tw2, adv); }
      if (tw2 && tw2.done) { clearInterval(this._iv); this.setState(x => ({ twOn: false, twN: x.twN + 1 })); } else this.setState(x => ({ twN: x.twN + 1 }));
    }, 500);
    this.setState({ twOn: true, twSrc: 'live' });
  }
  getFromSensors() {
    const L = this._live, f1 = (v, d) => Number(v).toFixed(d);
    if (!L) return;
    const age = L.lastWall ? (Date.now() - L.lastWall) / 1000 : null;
    if (age == null || age > 10) { this.setState({ sensMsg: 'No recent sensor data (' + (age == null ? 'no valid line received yet' : 'last line ' + f1(age, 0) + ' s ago') + '). Nothing was changed.' }); return; }
    const med = k => { const a = L.frames.slice(-5).map(r => r[k]).filter(v => isFinite(v)).sort((x, y) => x - y); return a.length ? a[a.length >> 1] : null; };
    const t1 = med('T1'), t2 = med('T2'), patch = {}, msg = [];
    if (t1 != null && t1 >= 1 && t1 <= 80) { patch.T0 = Math.round(t1 * 10) / 10; msg.push('Initial temperature = ' + f1(patch.T0, 1) + ' °C (T1, liquid probe)'); }
    else msg.push('Initial temperature NOT set: ' + (t1 == null ? 'T1 is missing' : 'T1 = ' + f1(t1, 1) + ' °C is outside 1–80'));
    if (t2 != null && t2 >= -10 && t2 <= 50) { patch.tAmb = Math.round(t2 * 10) / 10; msg.push('ambient = ' + f1(patch.tAmb, 1) + ' °C (T2)'); }
    else msg.push('ambient NOT set: ' + (t2 == null ? 'T2 is missing' : 'T2 = ' + f1(t2, 1) + ' °C is outside -10–50'));
    this.setState(s => ({ sc: { ...s.sc, ...patch }, sensMsg: msg.join('; ') + '. Median of the last 5 readings. T2 is the wall probe: read it before the run starts, while the wall is at room temperature.' }));
  }
  liveBits() {
    const L = this._live, st = this.state, f2 = (v, d) => (v == null || !isFinite(v) ? '–' : Number(v).toFixed(d));
    const ok = typeof navigator !== 'undefined' && 'serial' in navigator, R = st.runSc;
    const age = L && L.lastWall ? (Date.now() - L.lastWall) / 1000 : null, stale = L && (age == null || age > 10), fr = L && L.latest ? L.latest.fr : null;
    return {
      liveOn: !!L, sensConn: !!L, sensNotConn: !L, twSerial: () => this.serialConnect(), twDisconnect: () => this.disconnectLive(), twSerialDis: !ok, twSerialOp: ok ? 1 : 0.45,
      twSerialMsg: st.twErr || (!ok ? 'Browser serial needs desktop Chrome or Edge.' : ''),
      twBauds: [9600, 57600, 115200].map(b => ({ l: String(b), on: () => this.setState({ twBaud: b }), bg: (st.twBaud || 115200) === b ? BTN_ON : BTN_OFF, fg: (st.twBaud || 115200) === b ? 'oklch(0.18 0.01 250)' : C.wh })),
      getSens: () => this.getFromSensors(), sensMsg: st.sensMsg || '',
      sensStatus: L ? 'connected · ' + L.name + ' @ ' + L.baud + ' · ' + (fr ? 'T1 ' + f2(fr.T1, 1) + ' °C · T2 ' + f2(fr.T2, 1) + ' °C · last line ' + f2(age, 0) + ' s ago' : 'waiting for the first valid line…') : 'not connected',
      liveInfo: L ? L.name + ' @ ' + L.baud + ' baud · ' + L.nFrames + ' valid of ' + L.lines + ' lines · ' + L.gaps + ' gaps · ' + L.resets + ' resets' : '',
      twLiveNote: 'Assumed (not measured): volume noise max(3 mL, 3 %), liquid-T noise 0.5 °C, probe lag 15 s. t = 0 is the moment you press Start twin.',
      liveErr: L ? L.err : '', liveHasErr: !!(L && L.err),
      twDownload: () => { if (!L || !L.frames) return; const rows = ['t_s,T1_C,T2_C,P_bar_g,flow_L_min,I_A,V_V', ...L.frames.map(r => [r.t, r.T1, r.T2, r.P, r.flow, r.I, r.V].map(v => isFinite(v) ? v : 'nan').join(','))]; const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([rows.join('\n')], { type: 'text/csv' })); a.download = 'hydra_live_log.csv'; a.click(); },
      liveKpis: !L || !fr ? [] : [{ l: 'Liquid T (T1)', v: f2(fr.T1, 2) + ' °C', c: C.wh }, { l: 'Wall / ambient T (T2)', v: f2(fr.T2, 2) + ' °C', c: C.wh }, { l: 'Gauge pressure', v: f2(fr.P, 3) + ' bar', c: C.wh }, { l: 'H₂ flow', v: f2(fr.flow, 3) + ' L/min', c: C.wh },
        { l: 'Cumulative H₂ (integrated)', v: f2(L.latest.v, 1) + ' mL STP', c: C.cy }, { l: 'Logger time', v: f2(L.latest.t, 0) + ' s', c: C.wh }, { l: 'Last line', v: age == null ? '–' : f2(age, 0) + ' s ago', c: stale ? C.am : C.gr }],
      twStart: () => this.startTwinLive(), twStartDis: !R, twStartMsg: !R ? 'Run a scenario first (Scenario Builder → Run model): the twin uses that scenario\'s aluminium, electrolyte and vessel. Tip: use "Get from sensors" there to fill the temperatures.' : '',
    };
  }
  vTwin() {"""

BAUD_ROW = ('<span style="display:flex;gap:4px;align-items:center;font:10px ui-monospace,Menlo,Consolas,monospace;color:oklch(0.64 0.01 250)">baud'
            '<sc-for list="{{ twBauds }}" as="b"><button sc-camel-on-click="{{ b.on }}" style="background:{{ b.bg }};color:{{ b.fg }};border:1px solid oklch(0.36 0.01 250);font:600 11px ui-monospace,Menlo,Consolas,monospace;padding:6px 8px;cursor:pointer">{{ b.l }}</button></sc-for></span>')
BTN_STYLE = ("background:oklch(0.25 0.01 250);border:1px solid oklch(0.36 0.01 250);color:oklch(0.94 0.004 250);font:600 11px ui-monospace,Menlo,Consolas,monospace;"
             "letter-spacing:.05em;text-transform:uppercase;padding:7px 10px;cursor:pointer;white-space:nowrap")


def patch_template(t: str) -> str:
    assert "serialConnect" not in t, "template already patched"
    t = once(t, "twOn: false, twN: 0, twSrc: null,", "twOn: false, twN: 0, twSrc: null, twBaud: 115200, twErr: '', sensMsg: '',")
    t = once(t, "  vTwin() {", LIVE_METHODS)
    t = once(t, "    const o = {\n      twData: st.data.map(", "    const o = {\n      ...this.liveBits(),\n      twData: st.data.map(")
    t = once(t, "twStop: () => { clearInterval(this._iv); this.tw = null;", "twStop: () => { clearInterval(this._iv); this.closeLive(); this.tw = null;")
    t = once(t, "twIsData: !!tw && tw.src.kind === 'data',", "twIsData: !!tw && tw.src.kind === 'data', twIsLive: !!tw && tw.src.kind === 'live',")
    t = once(t, "(tw.done ? ' · finished' : st.twOn ? ' · running' : ' · paused') : '',",
             "(tw.done ? ' · finished' : st.twOn ? ' · running' : ' · paused') : (tw && tw.src.kind === 'live' ? (tw.src.latest ? 'assimilating…' : 'started, waiting for the next valid line…') : ''),")
    # Scenario page: sensor strip + same shared bits
    t = once(t, "const o = { segGroups, fields,", "const o = { ...this.liveBits(), segGroups, fields,")
    strip = ('<div style="border:1px solid oklch(0.33 0.01 250);padding:8px;display:flex;flex-direction:column;gap:6px">'
             '<div style="font:11px ui-monospace,Menlo,Consolas,monospace;color:oklch(0.80 0.006 250)">Sensors (Arduino over browser serial): {{ sensStatus }}</div>'
             '<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">'
             f'<sc-if value="{{{{ sensNotConn }}}}"><button sc-camel-on-click="{{{{ twSerial }}}}" disabled="{{{{ twSerialDis }}}}" style="{BTN_STYLE};opacity:{{{{ twSerialOp }}}}">Connect Arduino (browser serial)</button>{BAUD_ROW}</sc-if>'
             f'<sc-if value="{{{{ sensConn }}}}"><button sc-camel-on-click="{{{{ getSens }}}}" style="{BTN_STYLE}">Get initial + ambient temperature from sensors</button>'
             f'<button sc-camel-on-click="{{{{ twDisconnect }}}}" style="{BTN_STYLE}">Disconnect</button></sc-if></div>'
             '<div style="font:11px ui-monospace,Menlo,Consolas,monospace;color:oklch(0.80 0.11 75)">{{ twSerialMsg }}</div>'
             '<div style="font:11px ui-monospace,Menlo,Consolas,monospace;color:oklch(0.80 0.11 155)">{{ sensMsg }}</div></div>\n            ')
    t = once(t, '<sc-for list="{{ fields }}" as="f" hint-placeholder-count="13">', strip + '<sc-for list="{{ fields }}" as="f" hint-placeholder-count="13">')
    # Live Twin page markup
    t = once(t, "Live serial (Arduino/ESP32) and Modbus ingest need the local backend; the logger sketch is in Docs.",
             "Connect an Arduino/ESP32 logger over USB with browser serial (desktop Chrome or Edge): 7-column CSV at about 1 Hz, as sent by the hydra_logger sketch. "
             "Connect first, then press Start twin when the run begins: that moment is t = 0. Modbus ingest still needs the local backend.")
    m = re.search(r'<button disabled="\{\{ true \}\}" style="([^"]*)">Serial \(backend\)</button>', t)
    assert m, "serial button not found"
    style = m.group(1).replace("opacity:.45;cursor:not-allowed", "opacity:{{ twSerialOp }}")
    idle_btns = (f'<sc-if value="{{{{ sensNotConn }}}}"><button sc-camel-on-click="{{{{ twSerial }}}}" disabled="{{{{ twSerialDis }}}}" style="{style}">Connect Arduino (browser serial)</button>{BAUD_ROW}</sc-if>'
                 f'<sc-if value="{{{{ sensConn }}}}"><button sc-camel-on-click="{{{{ twStart }}}}" disabled="{{{{ twStartDis }}}}" style="{BTN_STYLE};opacity:1">Start twin from live data</button>'
                 f'<button sc-camel-on-click="{{{{ twDisconnect }}}}" style="{BTN_STYLE}">Disconnect</button></sc-if>')
    t = t.replace(m.group(0), idle_btns)
    t = once(t, '<sc-if value="{{ simNotReady }}">',
             '<div style="font:11px ui-monospace,Menlo,Consolas,monospace;color:oklch(0.80 0.11 75);max-width:640px">{{ twSerialMsg }}</div>\n'
             '            <sc-if value="{{ sensConn }}"><div style="font:11px ui-monospace,Menlo,Consolas,monospace;color:oklch(0.80 0.11 75);max-width:640px">{{ twStartMsg }}</div></sc-if>\n'
             '            <sc-if value="{{ simNotReady }}">')
    live_block = ('<sc-if value="{{ liveOn }}"><div style="display:flex;flex-direction:column;gap:6px">'
                  '<div style="font:700 11px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.06em;color:oklch(0.80 0.11 155);border:1px solid oklch(0.80 0.11 155);padding:6px 10px">LIVE serial · {{ liveInfo }}</div>'
                  '<sc-if value="{{ liveHasErr }}"><div style="font:700 11px ui-monospace,Menlo,Consolas,monospace;color:oklch(0.18 0.01 250);background:oklch(0.80 0.11 75);padding:6px 10px">{{ liveErr }}</div></sc-if>'
                  '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:1px;background:oklch(0.29 0.01 250);border:1px solid oklch(0.29 0.01 250)">'
                  '<sc-for list="{{ liveKpis }}" as="k"><div style="background:oklch(0.195 0.007 250);padding:6px 10px;display:flex;flex-direction:column;gap:2px"><span style="font:10px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.06em;text-transform:uppercase;color:oklch(0.66 0.01 250)">{{ k.l }}</span>'
                  '<span style="font:600 15px ui-monospace,Menlo,Consolas,monospace;color:{{ k.c }}">{{ k.v }}</span></div></sc-for></div>'
                  f'<div><button sc-camel-on-click="{{{{ twDownload }}}}" style="{BTN_STYLE}">Download received log (CSV)</button></div></div></sc-if>\n        '
                  '<sc-if value="{{ twIsLive }}"><div style="font:700 11px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.06em;color:oklch(0.80 0.11 155);border:1px solid oklch(0.80 0.11 155);padding:6px 10px">'
                  'MEASURED · LIVE · ensemble Kalman filter on integrated H₂ and liquid T</div>'
                  '<div style="font:11px ui-monospace,Menlo,Consolas,monospace;color:oklch(0.64 0.01 250)">{{ twLiveNote }}</div></sc-if>\n        ')
    t = once(t, '<sc-if value="{{ twIdle }}">', live_block + '<sc-if value="{{ twIdle }}">')
    t = once(t, "'EnKF replays uploaded measured CSVs. Serial/Modbus need the backend.'",
             "'EnKF replays uploaded measured CSVs, or assimilates a live Arduino/ESP32 over browser Web Serial (desktop Chrome/Edge); the Scenario page can read initial and ambient temperature from the sensors. Modbus needs the backend.'")
    return t


# ------------------------------------------------------------------------------------------------ bundle plumbing
def patch_page(page: Path) -> None:
    s = page.read_text()
    mt = '<script type="__bundler/manifest">'
    a = s.index(mt) + len(mt)
    b = s.index("</script>", a)
    man = json.loads(s[a:b])
    code = gzip.decompress(base64.b64decode(man[MODEL_ASSET]["data"])).decode()
    man[MODEL_ASSET]["data"] = base64.b64encode(gzip.compress(patch_model_js(code).encode(), 9)).decode()
    s = s[:a] + json.dumps(man, separators=(",", ":")) + s[b:]
    tt = '<script type="__bundler/template">'
    a = s.index(tt) + len(tt)
    b = s.index("</script>", a)
    tpl = patch_template(json.loads(s[a:b]))
    enc = json.dumps(tpl, ensure_ascii=False).replace("</", "<\\u002F")
    s = s[:a] + "\n" + enc + "\n  " + s[b:]
    page.write_text(s)


def main() -> None:
    page = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "prototype" / "hydra_l1_prototype.html"
    js = ROOT / "prototype" / "hydra_l1_model.js"
    js.write_text(patch_model_js(js.read_text()))
    patch_page(page)
    print("patched", page, "and", js)


if __name__ == "__main__":
    main()
