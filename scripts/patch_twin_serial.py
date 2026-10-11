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
// Lines are CSV. If the logger sends a header line (it must contain a time column such as t_s) columns are found BY NAME, so extra
// columns are fine; without a header the legacy order t_s,T1_C,T2_C,P_bar_g,flow_L_min,I_A,V_V is assumed. nan / blank = missing.
const LEGACY_COLS = ['t_s', 'T1_C', 'T2_C', 'P_bar_g', 'flow_L_min', 'I_A', 'V_V'];
const NA = new Set(['', 'nan', 'na', 'null']);
const isNum = x => x !== '' && isFinite(Number(x));
const pickCol = (cols, row, names, rx) => { for (const n of names) if (n in row) return n; if (rx) for (const c of cols) if (rx.test(c)) return c; return null; };
export function stdFrame(row, cols) {
  const get = k => (k && isFinite(row[k]) ? row[k] : NaN);
  const tc = pickCol(cols, row, ['t_s', 't_ms', 'time_s'], /^(t|time)(_s|_ms)?$/i);
  const tm = cols.filter(c => /(^|_)(t|temp|temperature)[0-9]*_?c$/i.test(c) && c !== tc);
  const t1 = 'T1_C' in row ? 'T1_C' : tm[0], t2 = 'T2_C' in row ? 'T2_C' : (tm.find(c => c !== t1) || null);
  return { t: get(tc) / (tc && /_ms$/i.test(tc) ? 1000 : 1), T1: get(t1), T2: get(t2), P: get(pickCol(cols, row, ['P_bar_g'], /^p.*bar/i)),
    flow: get(pickCol(cols, row, ['flow_L_min'], /flow.*l_?min/i)), I: get(pickCol(cols, row, ['I_A'])), V: get(pickCol(cols, row, ['V_V'])),
    mass: get(pickCol(cols, row, ['water_mass_g'], /mass.*_g$/i)), wet: get(pickCol(cols, row, ['H2_wet_ambient_mL'], /wet.*ml$/i)),
    vstp: get(pickCol(cols, row, [], /(stp|dry).*ml$/i)), row };
}
export function parseLine(src, line) {
  const s = line.trim(); if (!s || s[0] === '#') return null;
  const parts = s.split(',').map(x => x.trim());
  if (parts.length >= 2 && !isNum(parts[0]) && parts.every(p => /^[A-Za-z_][A-Za-z0-9_.\-()/%°]*$/.test(p)) && parts.some(p => /^(t|time)(_s|_ms)?$/i.test(p))) { src.cols = parts; return { header: parts }; }
  if (parts.length < 2) return null;
  const cols = src.cols || LEGACY_COLS, row = {};
  for (let i = 0; i < cols.length && i < parts.length; i++) { const x = parts[i].toLowerCase(); if (NA.has(x)) { row[cols[i]] = NaN; continue; } const n = Number(x); if (!isFinite(n)) return null; row[cols[i]] = n; }
  const fr = stdFrame(row, cols); if (!isFinite(fr.t)) return null;
  return { frame: fr };
}
export function parseFrame(line) { const r = parseLine({}, line); return r && r.frame ? r.frame : null; }
export function liveCreate() { return { kind: 'live', latest: null, tRel: 0, tPrev: null, cumL: 0, flowSeen: false, v0: null, conv: null, cols: null, frames: [], hist: [], console: [], nFrames: 0, lines: 0, gaps: 0, resets: 0, tauT: 15 }; }
// Gas collected over water is wet and at ambient conditions: dry STP volume = V * (P - p_w(T)) / 101325 * 273.15 / T (same formula as the dataset loader).
export function wetToStp(hPa, tcC) { const Ta = tcC + 273.15; return (hPa * 100 - psat(Ta)) / P_ATM * 273.15 / Ta; }
// Cumulative H2 [mL STP, dry] from the best available source: a dry/STP volume column, else the wet ambient column converted with p and tc
// (src.conv), else the integrated flow column. Column volumes are zeroed at the first frame after (re)start (src.v0).
function liveVolume(src, fr) {
  const colV = isFinite(fr.vstp) ? fr.vstp : (isFinite(fr.wet) && src.conv ? fr.wet * src.conv : null);
  if (colV != null) { if (src.v0 == null) src.v0 = colV; return colV - src.v0; }
  return src.flowSeen ? src.cumL * 1000 : NaN;
}
// Time is the logger's own clock (relative), so a stalled browser does not distort it; a logger reboot (t decreasing) is counted in .resets.
// Flow-integrated volume: gaps longer than 5 s are NOT integrated (counted in .gaps).
export function liveIngest(src, fr) {
  src.nFrames++;
  if (src.tPrev == null) src.tPrev = fr.t;
  const dt = fr.t - src.tPrev; src.tPrev = fr.t;
  if (dt < 0) src.resets++;
  else if (dt > 0) { src.tRel += dt; if (isFinite(fr.flow)) { src.flowSeen = true; if (dt > 5) src.gaps++; src.cumL += fr.flow * Math.min(dt, 5) / 60; } } // flow is integrated as reported (no clamping at 0: clamping turns zero-mean noise into a positive volume bias)
  if (src.frames.length < 100000) src.frames.push(fr);
  src.hist.push({ t: src.tRel, mass: fr.mass, wet: fr.wet, T1: fr.T1 }); if (src.hist.length > 3600) src.hist.shift();
  src.latest = { t: src.tRel, v: liveVolume(src, fr), T: fr.T1, fr };
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
                "export function twinTick(tw, adv = 20) {\n  if (tw.src.kind === 'live' && (!tw.src.latest || !isFinite(tw.src.latest.v))) return tw; // nothing received yet / no volume signal\n")
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
          let i; while ((i = buf.indexOf('\n')) >= 0) { const ln = buf.slice(0, i); buf = buf.slice(i + 1); if (ln.trim()) live.lines++; const res = M.parseLine(live, ln); if (res && res.frame) { M.liveIngest(live, res.frame); live.lastWall = Date.now(); } else if (res && res.header) live.console.push('[header] ' + res.header.join(',')); else if (ln.trim()) live.console.push(ln.trim()); if (live.console.length > 300) live.console.splice(0, live.console.length - 300); }
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
    live.cumL = 0; live.tRel = 0; live.latest = null; live.gaps = 0; live.resets = 0; live.v0 = null; // t = 0 and zero volume from this moment
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
  async sendCmd(text) {
    const L = this._live; text = String(text || '').trim();
    if (!L || !L.port || !L.port.writable || L.ended) return;
    if (!/^[A-Za-z0-9 .+\-_=]{1,40}$/.test(text)) { L.console.push('! not sent (letters, digits, space . + - _ = only, max 40 characters): ' + text); this.setState(x => ({ twN: x.twN + 1 })); return; }
    try {
      const w = L.port.writable.getWriter();
      try { await w.write(new TextEncoder().encode(text + '\n')); } finally { w.releaseLock(); }
      L.console.push('> ' + text);
      const m = /^(p|tc)\s+(-?[0-9.]+)$/i.exec(text);
      if (m) { L[m[1].toLowerCase() === 'p' ? 'hpa' : 'tcw'] = parseFloat(m[2]); if (isFinite(L.hpa) && isFinite(L.tcw)) { L.conv = this.M.wetToStp(L.hpa, L.tcw); L.v0 = null; } }
    } catch (e) { L.console.push('! send failed: ' + (e && e.message || e)); }
    if (L.console.length > 300) L.console.splice(0, L.console.length - 300);
    this.setState(x => ({ twN: x.twN + 1 }));
  }
  sendRanged(cmd, v, lo, hi, what) {
    const L = this._live; if (!L) return;
    if (!(isFinite(v) && v >= lo && v <= hi)) { L.console.push('! ' + what + ' must be between ' + lo + ' and ' + hi + ': not sent'); this.setState(x => ({ twN: x.twN + 1 })); return; }
    this.sendCmd(cmd + ' ' + v);
  }
  chkSet(kind) {
    const L = this._live; if (!L || !L.latest) return; const fr = L.latest.fr, v = kind === 'mass' ? fr.mass : fr.wet;
    L.chk = L.chk || {};
    L.chk[kind] = isFinite(v) ? { base: v, msg: 'Baseline set at ' + v.toFixed(1) + (kind === 'mass' ? ' g' : ' mL') + '. Now ' + (kind === 'mass' ? 'pour in the known amount of water' : 'push in the known volume of air') + ', let it settle, then press Check.' } : { base: null, msg: 'The logger sends no ' + (kind === 'mass' ? 'water_mass_g' : 'H2_wet_ambient_mL') + ' column.' };
    this.setState(x => ({ twN: x.twN + 1 }));
  }
  chkRun(kind) {
    const L = this._live, st = this.state; if (!L || !L.latest || !L.chk || !L.chk[kind] || L.chk[kind].base == null) return;
    const fr = L.latest.fr, amt = kind === 'mass' ? st.chkMass : st.chkGas, v = kind === 'mass' ? fr.mass : fr.wet, d = v - L.chk[kind].base;
    if (!isFinite(v) || !(amt > 0)) return;
    if (kind === 'mass') L.chk.mass.msg = 'Mass +' + d.toFixed(1) + ' g against ' + amt + ' g expected: ' + (Math.abs(d - amt) <= 2 ? 'PASS (within 2 g)' : 'FAIL (off by more than 2 g: recalibrate)');
    else { const r = d / amt; L.chk.gas.msg = 'Gas +' + d.toFixed(1) + ' mL = ' + (100 * r).toFixed(0) + ' % of ' + amt + ' mL expected: ' + (Math.abs(r - 1) <= 0.1 ? 'PASS (within 10 %)' : r < 0.9 ? 'FAIL (low: check every cap and tube joint for a leak)' : 'FAIL (high: check the baseline and the volume you pushed)'); }
    this.setState(x => ({ twN: x.twN + 1 }));
  }
  driftStart() { const L = this._live; if (!L || !L.latest) return; L.drift = { t0: L.latest.t }; this.setState(x => ({ twN: x.twN + 1 })); }
  liveBits() {
    const L = this._live, st = this.state, f2 = (v, d) => (v == null || !isFinite(v) ? '–' : Number(v).toFixed(d));
    const ok = typeof navigator !== 'undefined' && 'serial' in navigator, R = st.runSc;
    const age = L && L.lastWall ? (Date.now() - L.lastWall) / 1000 : null, stale = L && (age == null || age > 10), fr = L && L.latest ? L.latest.fr : null;
    return {
      liveOn: !!L, sensConn: !!L, sensNotConn: !L, idleTitle: L ? 'Logger connected: calibrate it in the control panel above, then start the twin.' : 'No sensor connected — upload a CSV or connect a logger.', twSerial: () => this.serialConnect(), twDisconnect: () => this.disconnectLive(), twSerialDis: !ok, twSerialOp: ok ? 1 : 0.45,
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
      ...(() => {
        const dis = !L || L.ended, H = L ? L.hist : [], last = H.length ? H[H.length - 1] : null;
        const w30 = last ? H.filter(h => h.t >= last.t - 30 && isFinite(h.mass)).map(h => h.mass) : [];
        const num = (key, label, unit, btn, go) => ({ label, unit, val: st[key] == null ? '' : st[key], on: e => { const r = e.target.value; this.setState({ [key]: r === '' ? null : parseFloat(r) }); }, hasInput: true, b1: btn, g1: go, dis1: dis, hasB2: false, b2: '', g2: () => { }, dis2: true, msg: '', mc: C.mu });
        const mk = (kind, label, unit, key, name) => { const c = L && L.chk && L.chk[kind], msg = c ? c.msg : '';
          return { label, unit, val: st[key] == null ? '' : st[key], on: e => { const r = e.target.value; this.setState({ [key]: r === '' ? null : parseFloat(r) }); }, hasInput: true, b1: 'Set baseline', g1: () => this.chkSet(kind), dis1: dis || !L.latest,
            hasB2: true, b2: 'Check', g2: () => this.chkRun(kind), dis2: dis || !(c && c.base != null), msg, mc: /FAIL/.test(msg) ? C.am : /PASS/.test(msg) ? C.gr : C.mu }; };
        let dmsg = '';
        if (L && L.drift) { const el = L.latest ? L.latest.t - L.drift.t0 : 0, w = H.filter(h => h.t >= L.drift.t0 && isFinite(h.mass)).map(h => h.mass);
          if (!w.length) dmsg = 'No water_mass_g readings yet.'; else { const rg = Math.max(...w) - Math.min(...w);
            dmsg = el < 300 ? 'Running: ' + f2(el, 0) + ' / 300 s, range so far ' + f2(rg, 2) + ' g (leave the empty beaker untouched)' : 'Done: range ' + f2(rg, 2) + ' g: ' + (rg <= 1.0 ? 'PASS (within ±0.5 g)' : 'FAIL (more than ±0.5 g: check the mounting screws and wires)'); } }
        const drift = { label: 'Steady test, empty beaker, 5 min', unit: '', val: '', on: () => { }, hasInput: false, b1: 'Start steady test', g1: () => this.driftStart(), dis1: dis || !(L && L.latest), hasB2: false, b2: '', g2: () => { }, dis2: true, msg: dmsg, mc: /FAIL/.test(dmsg) ? C.am : /PASS/.test(dmsg) ? C.gr : C.mu };
        const rng = (key, cmd, lo, hi, what) => () => this.sendRanged(cmd, st[key], lo, hi, what);
        return {
          calBtns: [['Tare', 'tare'], ['Start logging', 'start'], ['Stop logging', 'stop']].map(([l, c]) => ({ l, go: () => this.sendCmd(c), dis })),
          calRows: [num('calKnown', 'Known weight on the platform', 'g', 'Calibrate (cal)', rng('calKnown', 'cal', 1, 5000, 'known weight (g)')), num('calHpa', 'Ambient pressure today', 'hPa', 'Set p', rng('calHpa', 'p', 800, 1100, 'pressure (hPa)')), num('calTc', 'Water temperature', '°C', 'Set tc', rng('calTc', 'tc', -10, 100, 'water temperature (°C)'))],
          chkRows: [mk('mass', 'Mass check: water poured in', 'g', 'chkMass'), mk('gas', 'Gas check: air pushed in', 'mL', 'chkGas'), drift],
          cmdVal: st.cmdText || '', cmdOn: e => this.setState({ cmdText: e.target.value }), cmdSend: () => { this.sendCmd(st.cmdText); this.setState({ cmdText: '' }); }, cmdDis: dis,
          consoleText: L ? L.console.slice(-40).join('\n') : '',
          convNote: L && L.conv ? 'Wet-gas to dry-STP factor ' + f2(L.conv, 4) + ' (p = ' + L.hpa + ' hPa, water ' + L.tcw + ' °C). Used for the twin when the logger sends H2_wet_ambient_mL but no dry-STP column.' : 'Send p and tc so the twin can convert H2_wet_ambient_mL to dry STP (not needed if the logger sends a dry-STP column or flow_L_min).',
          calKpis: !L || !last ? [] : [{ l: 'Load cell (water_mass_g)', v: f2(last.mass, 2) + ' g', c: C.cy }, { l: 'Mass range, last 30 s', v: w30.length > 1 ? f2(Math.max(...w30) - Math.min(...w30), 2) + ' g' : '–', c: C.wh }, { l: 'Gas, wet ambient', v: f2(last.wet, 1) + ' mL', c: C.wh }, { l: 'Probe temperature', v: f2(last.T1, 2) + ' °C', c: C.wh },
            { l: 'Columns seen', v: L.cols ? String(L.cols.length) : '7 (legacy)', c: C.mu }],
          colNames: L && L.cols ? L.cols.join(', ') : '',
        };
      })(),
      twStart: () => this.startTwinLive(), twStartDis: !R, twStartMsg: !R ? 'Run a scenario first (Scenario Builder → Run model): the twin uses that scenario\'s aluminium, electrolyte and vessel. Tip: use "Get from sensors" there to fill the temperatures.' : '',
    };
  }
  vTwin() {"""

BAUD_ROW = ('<span style="display:flex;gap:4px;align-items:center;font:10px ui-monospace,Menlo,Consolas,monospace;color:oklch(0.64 0.01 250)">baud'
            '<sc-for list="{{ twBauds }}" as="b"><button sc-camel-on-click="{{ b.on }}" style="background:{{ b.bg }};color:{{ b.fg }};border:1px solid oklch(0.36 0.01 250);font:600 11px ui-monospace,Menlo,Consolas,monospace;padding:6px 8px;cursor:pointer">{{ b.l }}</button></sc-for></span>')
BTN_STYLE = ("background:oklch(0.25 0.01 250);border:1px solid oklch(0.36 0.01 250);color:oklch(0.94 0.004 250);font:600 11px ui-monospace,Menlo,Consolas,monospace;"
             "letter-spacing:.05em;text-transform:uppercase;padding:7px 10px;cursor:pointer;white-space:nowrap")


MONO = "font:11px ui-monospace,Menlo,Consolas,monospace"
INPUT_STYLE = "background:oklch(0.15 0.006 250);border:1px solid oklch(0.33 0.01 250);color:oklch(0.94 0.004 250);font:12px ui-monospace,Menlo,Consolas,monospace;padding:4px 6px;width:100%;box-sizing:border-box"
PANEL = (
    '<div style="border:1px solid oklch(0.33 0.01 250);padding:10px;display:flex;flex-direction:column;gap:8px">'
    f'<div style="font:600 11px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase;color:oklch(0.86 0.005 250)">Logger control panel · calibrate and check before any chemicals</div>'
    f'<div style="{MONO};color:oklch(0.64 0.01 250)">Sends plain text commands to the logger (tare, cal, p, tc, start, stop) only when you press a button. Nothing is sent automatically. Columns seen: {{{{ colNames }}}}</div>'
    '<div style="display:flex;gap:6px;flex-wrap:wrap"><sc-for list="{{ calBtns }}" as="c">'
    f'<button sc-camel-on-click="{{{{ c.go }}}}" disabled="{{{{ c.dis }}}}" style="{BTN_STYLE}">{{{{ c.l }}}}</button></sc-for></div>'
    '<sc-for list="{{ calRows }}" as="r"><div style="display:grid;grid-template-columns:minmax(0,1fr) 90px 150px;gap:8px;align-items:center">'
    f'<span style="{MONO};color:oklch(0.80 0.006 250)">{{{{ r.label }}}} <span style="color:oklch(0.60 0.01 250)">{{{{ r.unit }}}}</span></span>'
    f'<input type="number" step="any" value="{{{{ r.val }}}}" sc-camel-on-change="{{{{ r.on }}}}" style="{INPUT_STYLE}">'
    f'<button sc-camel-on-click="{{{{ r.g1 }}}}" disabled="{{{{ r.dis1 }}}}" style="{BTN_STYLE}">{{{{ r.b1 }}}}</button></div></sc-for>'
    f'<div style="{MONO};color:oklch(0.64 0.01 250)">{{{{ convNote }}}}</div>'
    '<div style="font:600 11px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase;color:oklch(0.86 0.005 250)">Checks</div>'
    '<sc-for list="{{ chkRows }}" as="r"><div style="display:flex;flex-direction:column;gap:4px">'
    '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
    f'<span style="{MONO};color:oklch(0.80 0.006 250);min-width:230px">{{{{ r.label }}}}</span>'
    '<sc-if value="{{ r.hasInput }}">'
    f'<input type="number" step="any" value="{{{{ r.val }}}}" sc-camel-on-change="{{{{ r.on }}}}" style="{INPUT_STYLE};width:80px"><span style="{MONO};color:oklch(0.60 0.01 250)">{{{{ r.unit }}}}</span></sc-if>'
    f'<button sc-camel-on-click="{{{{ r.g1 }}}}" disabled="{{{{ r.dis1 }}}}" style="{BTN_STYLE}">{{{{ r.b1 }}}}</button>'
    f'<sc-if value="{{{{ r.hasB2 }}}}"><button sc-camel-on-click="{{{{ r.g2 }}}}" disabled="{{{{ r.dis2 }}}}" style="{BTN_STYLE}">{{{{ r.b2 }}}}</button></sc-if></div>'
    f'<div style="{MONO};color:{{{{ r.mc }}}}">{{{{ r.msg }}}}</div></div></sc-for>'
    '<div style="display:flex;gap:6px;align-items:center">'
    f'<input type="text" value="{{{{ cmdVal }}}}" sc-camel-on-change="{{{{ cmdOn }}}}" placeholder="other command, e.g. tc 22" style="{INPUT_STYLE};max-width:320px">'
    f'<button sc-camel-on-click="{{{{ cmdSend }}}}" disabled="{{{{ cmdDis }}}}" style="{BTN_STYLE}">Send</button></div>'
    f'<div style="{MONO};color:oklch(0.64 0.01 250)">Logger messages (newest last)</div>'
    f'<div style="{MONO};white-space:pre-wrap;background:oklch(0.15 0.006 250);border:1px solid oklch(0.29 0.01 250);padding:6px 8px;height:150px;overflow:auto;color:oklch(0.86 0.005 250)">{{{{ consoleText }}}}</div>'
    '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:1px;background:oklch(0.29 0.01 250);border:1px solid oklch(0.29 0.01 250)">'
    '<sc-for list="{{ calKpis }}" as="k"><div style="background:oklch(0.195 0.007 250);padding:6px 10px;display:flex;flex-direction:column;gap:2px">'
    '<span style="font:10px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.06em;text-transform:uppercase;color:oklch(0.66 0.01 250)">{{ k.l }}</span>'
    '<span style="font:600 15px ui-monospace,Menlo,Consolas,monospace;color:{{ k.c }}">{{ k.v }}</span></div></sc-for></div></div>'
)



def patch_template(t: str) -> str:
    assert "serialConnect" not in t, "template already patched"
    t = once(t, "twOn: false, twN: 0, twSrc: null,", "twOn: false, twN: 0, twSrc: null, twBaud: 115200, twErr: '', sensMsg: '', calKnown: 200, calHpa: null, calTc: null, chkMass: 100, chkGas: 60, cmdText: '',")
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
             "Connect an Arduino/ESP32 logger over USB with browser serial (desktop Chrome or Edge). Columns are read by name from the logger's header line (t_s, T1_C, water_mass_g, H2_wet_ambient_mL, flow_L_min, …); with no header the 7-column hydra_logger order is assumed. "
             "Connect first, then press Start twin when the run begins: that moment is t = 0. Modbus ingest still needs the local backend.")
    m = re.search(r'<button disabled="\{\{ true \}\}" style="([^"]*)">Serial \(backend\)</button>', t)
    assert m, "serial button not found"
    style = m.group(1).replace("opacity:.45;cursor:not-allowed", "opacity:{{ twSerialOp }}")
    idle_btns = (f'<sc-if value="{{{{ sensNotConn }}}}"><button sc-camel-on-click="{{{{ twSerial }}}}" disabled="{{{{ twSerialDis }}}}" style="{style}">Connect Arduino (browser serial)</button>{BAUD_ROW}</sc-if>'
                 f'<sc-if value="{{{{ sensConn }}}}"><button sc-camel-on-click="{{{{ twStart }}}}" disabled="{{{{ twStartDis }}}}" style="{BTN_STYLE};opacity:1">Start twin from live data</button>'
                 f'<button sc-camel-on-click="{{{{ twDisconnect }}}}" style="{BTN_STYLE}">Disconnect</button></sc-if>')
    t = t.replace(m.group(0), idle_btns)
    t = once(t, '">No sensor connected — upload a CSV or connect a logger.</div>', '">{{ idleTitle }}</div>')
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
                  f'<div><button sc-camel-on-click="{{{{ twDownload }}}}" style="{BTN_STYLE}">Download received log (CSV)</button></div>' + PANEL + '</div></sc-if>\n        '
                  '<sc-if value="{{ twIsLive }}"><div style="font:700 11px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.06em;color:oklch(0.80 0.11 155);border:1px solid oklch(0.80 0.11 155);padding:6px 10px">'
                  'MEASURED · LIVE · ensemble Kalman filter on integrated H₂ and liquid T</div>'
                  '<div style="font:11px ui-monospace,Menlo,Consolas,monospace;color:oklch(0.64 0.01 250)">{{ twLiveNote }}</div></sc-if>\n        ')
    t = once(t, '<sc-if value="{{ twIdle }}">', live_block + '<sc-if value="{{ twIdle }}">')
    t = once(t, "'EnKF replays uploaded measured CSVs. Serial/Modbus need the backend.'",
             "'EnKF replays uploaded measured CSVs, or assimilates a live Arduino/ESP32 over browser Web Serial (desktop Chrome/Edge); the Scenario page can read initial and ambient temperature from the sensors; the Live Twin has a logger control panel (tare, cal, p, tc, start, stop, checks). Modbus needs the backend.'")
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
