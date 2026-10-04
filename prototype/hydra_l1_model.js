// HYDRA L1 lumped model of Al–NaOH–H2O hydrogen generation + AEM stack (browser prototype).
// SI internally except where a field says otherwise. Every parameter lives in PARAMS with unit, range and source placeholder.
// Status of every default: UNCALIBRATED PRIOR. Do not treat as fact.
// This file contains no scenarios, no experimental data and no example datasets.
// VERIFY_SC below is a code-verification fixture used only by the Docs checks.

export const R = 8.314462618, F = 96485.33212, P_ATM = 101325, VM = 22.414, LHV = 241.8e3;
export const MW = { Al: 26.9815, NaOH: 39.997, H2O: 18.015, H2: 2.016 };

export const PARAMS = {
  ks25:  { g: 'Kinetics', sym: 'k_s,25', label: 'Surface rate constant at 25 °C', unit: 'mol cm⁻² s⁻¹ M⁻ⁿ', def: 2.0e-8, lo: 5e-9, hi: 1e-7, dist: 'logn', sd: 0.6, src: '[SRC-K1]' },
  Ea:    { g: 'Kinetics', sym: 'E_a', label: 'Apparent activation energy', unit: 'J mol⁻¹', def: 45e3, lo: 35e3, hi: 70e3, dist: 'norm', sd: 6e3, src: '[SRC-K2]' },
  n:     { g: 'Kinetics', sym: 'n', label: 'Apparent reaction order in OH⁻', unit: '–', def: 0.8, lo: 0.5, hi: 1.2, dist: 'norm', sd: 0.1, src: '[SRC-K3]' },
  tauInd:{ g: 'Kinetics', sym: 'τ_ind', label: 'Oxide-film induction time constant', unit: 's', def: 60, lo: 10, hi: 600, dist: 'logn', sd: 0.5, src: '[SRC-K4]' },
  km:    { g: 'Transport', sym: 'k_m', label: 'OH⁻ mass-transfer coefficient (Da only)', unit: 'cm s⁻¹', def: 5e-3, lo: 1e-3, hi: 2e-2, dist: 'logn', sd: 0.5, src: '[SRC-T1]' },
  dH:    { g: 'Thermo', sym: 'ΔH_r', label: 'Reaction enthalpy per mol Al', unit: 'J mol⁻¹', def: -415e3, lo: -430e3, hi: -405e3, dist: 'norm', sd: 4e3, src: '[SRC-H1]' },
  cpLiq: { g: 'Thermo', sym: 'c_p,l', label: 'Electrolyte specific heat', unit: 'J g⁻¹ K⁻¹', def: 3.75, lo: 3.55, hi: 3.95, dist: 'norm', sd: 0.06, src: '[SRC-H2]' },
  rhoLiq:{ g: 'Thermo', sym: 'ρ_l', label: 'Electrolyte density (fixed, not f(c,T))', unit: 'g mL⁻¹', def: 1.08, lo: 1.04, hi: 1.12, dist: null, src: '[SRC-H3]' },
  aw:    { g: 'Thermo', sym: 'a_w', label: 'Water activity (vapour-pressure factor)', unit: '–', def: 0.93, lo: 0.88, hi: 0.98, dist: null, src: '[SRC-H4]' },
  hfg:   { g: 'Thermo', sym: 'h_fg', label: 'Latent heat of vaporisation', unit: 'J g⁻¹', def: 2257, lo: 2200, hi: 2442, dist: null, src: 'steam tables [SRC-H5]' },
  UA:    { g: 'Vessel', sym: 'UA', label: 'Overall heat-loss conductance', unit: 'W K⁻¹', def: 0.45, lo: 0.2, hi: 1.5, dist: 'logn', sd: 0.3, src: 'estimate [SRC-V1]' },
  mVes:  { g: 'Vessel', sym: 'm_v', label: 'HDPE vessel mass', unit: 'g', def: 35, lo: 25, hi: 60, dist: null, src: 'measure' },
  cpVes: { g: 'Vessel', sym: 'c_p,v', label: 'HDPE specific heat', unit: 'J g⁻¹ K⁻¹', def: 1.9, lo: 1.8, hi: 2.3, dist: null, src: '[SRC-V2]' },
  rVes:  { g: 'Vessel', sym: 'r', label: 'Vessel inner radius', unit: 'm', def: 0.04, lo: 0.03, hi: 0.06, dist: null, src: 'measure' },
  tVes:  { g: 'Vessel', sym: 't_w', label: 'Wall thickness', unit: 'm', def: 0.0015, lo: 0.001, hi: 0.003, dist: null, src: 'measure' },
  sy23:  { g: 'Vessel', sym: 'σ_y,23', label: 'HDPE yield strength at 23 °C', unit: 'MPa', def: 26, lo: 20, hi: 32, dist: null, src: '[SRC-V3]' },
  dsy:   { g: 'Vessel', sym: 'dσ_y/dT', label: 'Yield-strength temperature slope', unit: 'MPa K⁻¹', def: 0.25, lo: 0.15, hi: 0.35, dist: null, src: '[SRC-V4]' },
  THdpe: { g: 'Vessel', sym: 'T_max', label: 'HDPE max service temperature', unit: '°C', def: 80, lo: 60, hi: 110, dist: null, src: '[SRC-V5]' },
  i0a:   { g: 'Fuel cell', sym: 'i_0,a', label: 'NiMo HOR exchange current density', unit: 'A cm⁻²', def: 5e-3, lo: 1e-4, hi: 2e-2, dist: 'logn', sd: 0.8, src: '[SRC-F1]' },
  aa:    { g: 'Fuel cell', sym: 'α_a', label: 'Anode transfer coefficient', unit: '–', def: 0.5, lo: 0.3, hi: 1.0, dist: null, src: '[SRC-F2]' },
  i0c:   { g: 'Fuel cell', sym: 'i_0,c', label: 'Ag ORR exchange current density', unit: 'A cm⁻²', def: 1e-4, lo: 1e-6, hi: 1e-3, dist: 'logn', sd: 0.8, src: '[SRC-F3]' },
  ac:    { g: 'Fuel cell', sym: 'α_c', label: 'Cathode transfer coefficient', unit: '–', def: 0.8, lo: 0.4, hi: 1.2, dist: null, src: '[SRC-F4]' },
  asr:   { g: 'Fuel cell', sym: 'ASR', label: 'Area-specific resistance (homemade AEM)', unit: 'Ω cm²', def: 0.30, lo: 0.1, hi: 0.8, dist: 'logn', sd: 0.3, src: '[SRC-F5]' },
  iL:    { g: 'Fuel cell', sym: 'i_L', label: 'Limiting current density', unit: 'A cm⁻²', def: 0.6, lo: 0.3, hi: 1.2, dist: null, src: '[SRC-F6]' },
  Tfc:   { g: 'Fuel cell', sym: 'T_fc', label: 'Stack temperature (held constant)', unit: 'K', def: 323.15, lo: 298.15, hi: 343.15, dist: null, src: 'operating choice' },
};

export const FORMS = {
  foil:   { label: 'Foil', dim: 'Thickness', unit: 'µm', min: 5, max: 200, step: 5, g: 0.15, active: 1, sa: d => 2 / (2.70 * d * 1e-4) },
  powder: { label: 'Powder', dim: 'Diameter', unit: 'µm', min: 10, max: 500, step: 10, g: 2 / 3, active: 1, sa: d => 6 / (2.70 * d * 1e-4) },
  wire:   { label: 'Wire', dim: 'Diameter', unit: 'µm', min: 200, max: 3000, step: 100, g: 0.5, active: 1, sa: d => 4 / (2.70 * d * 1e-4) },
  can:    { label: 'Crushed can', dim: 'Wall', unit: 'µm', min: 50, max: 200, step: 10, g: 0.15, active: 0.6, sa: d => 2 / (2.70 * d * 1e-4) },
};
export const ALLOYS = {
  pure:  { label: 'Pure Al', mult: 1.0, src: 'reference' },
  a1xxx: { label: '1xxx', mult: 0.95, src: '[SRC-A1] fit-able' },
  a3xxx: { label: '3xxx (can)', mult: 0.8, src: '[SRC-A2] fit-able' },
  a6061: { label: '6061', mult: 0.7, src: '[SRC-A3] fit-able' },
};


export function defaults() { const o = {}; for (const k in PARAMS) o[k] = PARAMS[k].def; return o; }
export function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
export function randn(rng) { let u = 0; while (u === 0) u = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()); }
export function drawParams(base, rng, keys) {
  const p = { ...base };
  for (const k of keys || Object.keys(PARAMS)) {
    const d = PARAMS[k]; if (!d.dist) continue;
    const z = randn(rng); const v = d.dist === 'logn' ? base[k] * Math.exp(d.sd * z) : base[k] + d.sd * z;
    p[k] = Math.min(Math.max(d.lo, d.hi), Math.max(Math.min(d.lo, d.hi), v));
  }
  return p;
}

// Antoine (water, mmHg, °C), two ranges
export function psat(T) { const c = T - 273.15; const [A, B, C] = c < 100 ? [8.07131, 1730.63, 233.426] : [8.14019, 1810.94, 244.485]; return 133.322 * Math.pow(10, A - B / (C + c)); }
export function tBoil(P) { const l = Math.log10(P / 133.322); let c = 1730.63 / (8.07131 - l) - 233.426; if (c > 100) c = 1810.94 / (8.14019 - l) - 244.485; return c + 273.15; }

export function loadI(sc, t) {
  const A = sc.cellArea;
  switch (sc.load) {
    case 'step': return A * (t < 600 ? 0.05 : t < 2400 ? 0.2 : 0.1);
    case 'const': return A * 0.15;
    case 'ramp': return A * 0.25 * Math.min(t / sc.duration, 1);
    default: return 0;
  }
}

export function polar(i, p, pH2) {
  const T = p.Tfc;
  const E = 1.229 - 8.5e-4 * (T - 298.15) + R * T / (2 * F) * Math.log(Math.max(pH2, 1e-3) * Math.sqrt(0.21));
  if (i <= 0) return { V: E, E, ea: 0, ec: 0, eo: 0, en: 0 };
  const ea = R * T / (p.aa * F) * Math.asinh(i / (2 * p.i0a));
  const ec = R * T / (p.ac * F) * Math.asinh(i / (2 * p.i0c));
  const eo = i * p.asr; const en = i < p.iL ? -0.03 * Math.log(1 - i / p.iL) : 1;
  return { V: Math.max(E - ea - ec - eo - en, 0), E, ea, ec, eo, en };
}

export function init(sc, p) {
  const f = FORMS[sc.form], T = sc.T0 + 273.15, Vl = sc.vLiq, Vh = Math.max(sc.vVessel - Vl, 1) * 1e-6;
  const nAl = sc.alMass / MW.Al, nOH = sc.cNaOH * Vl / 1000;
  return { t: 0, nAl0: nAl, nAl, A0: sc.alMass * f.sa(sc.dim) * f.active, nOH, nAlu: 0, Vl, T,
    nAir: Math.max(P_ATM - p.aw * psat(T), 0) * Vh / (R * T), nH2: 0, gen: 0, vent: 0, stack: 0, purge: 0, evap: 0,
    Wel: 0, Qrx: 0, Qloss: 0, nRoom: 0, r: 0, P: P_ATM, I: 0, V: 0, u: 0, dosed: 0, boil: false, starve: false, tStarve: 0, psi: 0, Da: 0 };
}

export function step(s, sc, p, dt, u = 0, cDose = 0) {
  const f = FORMS[sc.form], al = ALLOYS[sc.alloy];
  s.u = u; if (u > 0) { s.nOH += u * dt * cDose / 1000; s.Vl += u * dt; s.dosed += u * dt; }
  const c = Math.max(s.nOH / (s.Vl * 1e-3), 0);
  const fI = 1 - Math.exp(-s.t / p.tauInd);
  const A = s.A0 * Math.pow(Math.max(s.nAl / s.nAl0, 0), f.g);
  const k = p.ks25 * al.mult * Math.exp(-p.Ea / R * (1 / s.T - 1 / 298.15));
  let r = k * A * Math.pow(c, p.n) * fI;
  r = Math.max(0, Math.min(r, s.nAl / dt, s.nOH / dt));
  s.nAl -= r * dt; s.nOH -= r * dt; s.nAlu += r * dt; s.r = r;
  const g = 1.5 * r * dt; s.nH2 += g; s.gen += g;
  const Qr = -p.dH * r, Ql = p.UA * (s.T - sc.tAmb - 273.15);
  const C = s.Vl * p.rhoLiq * p.cpLiq + p.mVes * p.cpVes;
  let T = s.T + (Qr - Ql) * dt / C;
  s.Qrx += Qr * dt; s.Qloss += Ql * dt;
  const Plim = sc.mode === 'open' ? P_ATM : sc.mode === 'relief' ? P_ATM + sc.pRelief * 1e5 : Infinity;
  s.boil = false;
  if (Plim < Infinity) {
    const Tb = tBoil(Plim / p.aw);
    if (T > Tb) { const m = (T - Tb) * C / p.hfg; s.evap += m; s.Vl = Math.max(s.Vl - m / p.rhoLiq, 1); T = Tb; s.boil = true; }
  }
  s.T = T;
  const Vh = Math.max(sc.vVessel - s.Vl, 1) * 1e-6, pv = p.aw * psat(T), RT = R * T;
  let nNC = s.nAir + s.nH2;
  const Idem = sc.mode === 'open' || !sc.cells ? 0 : loadI(sc, s.t);
  s.I = 0; s.starve = false;
  if (Idem > 0) {
    const dem = Idem * sc.cells / (2 * F) * dt;
    const P = nNC * RT / Vh + pv;
    const nMin = Math.max(P_ATM + 5e3 - pv, 0) * Vh / RT, avail = P > P_ATM + 5e3 ? Math.max(nNC - nMin, 0) : 0;
    const x = s.nH2 / (nNC || 1e-30);
    if (x >= 0.9 && avail > 0) {
      const take = Math.min(dem, avail * x);
      s.nH2 -= take; s.nAir = Math.max(s.nAir - take * (1 - x) / x, 0); s.stack += take; s.I = Idem * take / dem; s.starve = take < 0.98 * dem;
    } else {
      const tot = Math.min(3e-4 * dt, avail);
      s.nH2 -= tot * x; s.nAir = Math.max(s.nAir - tot * (1 - x), 0); s.purge += tot * x; s.nRoom += tot * x; s.starve = true;
    }
    if (s.starve) s.tStarve += dt;
    nNC = s.nAir + s.nH2;
  }
  if (nNC * RT / Vh + pv > Plim) {
    const allow = Math.max(Plim - pv, 0) * Vh / RT, ex = nNC - allow;
    if (ex > 0) { const x = s.nH2 / nNC; s.nH2 -= ex * x; s.nAir = Math.max(s.nAir - ex * (1 - x), 0); s.vent += ex * x; if (sc.mode !== 'open') s.nRoom += ex * x; }
  }
  s.nRoom -= s.nRoom * sc.ach / 3600 * dt;
  s.P = (s.nAir + s.nH2) * RT / Vh + pv;
  if (sc.cells) { const pol = polar(s.I / sc.cellArea, p, Math.max(s.P / 1e5, 1)); s.V = pol.V * sc.cells; s.Wel += s.V * s.I * dt; }
  s.psi = (Qr * p.Ea / (R * T * T)) / p.UA;
  s.Da = k * 1000 * Math.pow(Math.max(c, 1e-3), p.n - 1) / p.km;
  s.t += dt;
}

export function sf(s, p) { const Pg = Math.max(s.P - P_ATM, 0); if (Pg < 1) return 99; const sig = Pg * p.rVes / p.tVes / 1e6; return Math.min(Math.max(p.sy23 - p.dsy * (s.T - 296.15), 2) / sig, 99); }
export const roomPct = (s, sc) => s.nRoom * R * 298.15 / P_ATM / sc.roomVol * 100;
export const KEYS = ['t', 'h2', 'h2act', 'flow', 'dem', 'T', 'P', 'c', 'alu', 'alPct', 'Vs', 'Ps', 'vent', 'room', 'sf', 'Da', 'psi', 'u'];

export function simulate(sc, p, opt = {}) {
  const dt = opt.dt || 2, every = opt.every || 5, N = Math.round(sc.duration / dt);
  const s = opt.state || init(sc, p);
  const o = {}; KEYS.forEach(k => o[k] = []);
  const ev = { tVent: null, tBoil: null, tHdpe: null, tRoom: null, tSF: null };
  const pk = { flow: 0, T: -1e9, P: 0, room: 0, minSF: 99 };
  const theoAl = 1.5 * s.nAl0 * VM; let t90 = null;
  const rec = () => {
    const L = s.Vl * 1e-3;
    o.t.push(s.t / 60); o.h2.push(s.gen * VM); o.h2act.push(s.gen * R * (sc.tAmb + 273.15) / P_ATM * 1000);
    o.flow.push(s.r * 1.5 * VM * 6e4); o.dem.push(sc.mode === 'open' || !sc.cells ? 0 : loadI(sc, s.t) * sc.cells / (2 * F) * VM * 6e4);
    o.T.push(s.T - 273.15); o.P.push(s.P / 1e5); o.c.push(s.nOH / L); o.alu.push(s.nAlu / L); o.alPct.push(100 * s.nAl / s.nAl0);
    o.Vs.push(s.V); o.Ps.push(s.V * s.I); o.vent.push((s.vent + s.purge) * VM); o.room.push(roomPct(s, sc)); o.sf.push(sf(s, p));
    o.Da.push(s.Da); o.psi.push(s.psi); o.u.push(s.u * 60);
  };
  rec();
  for (let i = 0; i < N; i++) {
    const u = opt.ctrl ? opt.ctrl(s) : 0;
    step(s, sc, p, dt, u, opt.cDose || 0);
    const tm = s.t / 60, fl = s.r * 1.5 * VM * 6e4, Tc = s.T - 273.15, rp = roomPct(s, sc), S = sf(s, p);
    if (fl > pk.flow) pk.flow = fl; if (Tc > pk.T) pk.T = Tc; if (s.P > pk.P) pk.P = s.P; if (rp > pk.room) pk.room = rp; if (S < pk.minSF) pk.minSF = S;
    if (ev.tVent == null && s.vent > 1e-6 && sc.mode !== 'open') ev.tVent = tm;
    if (ev.tBoil == null && s.boil) ev.tBoil = tm;
    if (ev.tHdpe == null && Tc >= p.THdpe) ev.tHdpe = tm;
    if (ev.tRoom == null && rp >= 1) ev.tRoom = tm;
    if (ev.tSF == null && S < 2) ev.tSF = tm;
    if (t90 == null && s.gen * VM >= 0.9 * theoAl) t90 = tm;
    if ((i + 1) % every === 0) rec();
  }
  const reacted = s.nAl0 - s.nAl;
  const sum = {
    total: s.gen * VM, theoAl, theo: 1.5 * Math.min(s.nAl0, init(sc, p).nOH) * VM, pct: 100 * s.gen * VM / theoAl,
    peakFlow: pk.flow, peakT: pk.T, peakP: pk.P / 1e5, t90, Wh: s.Wel / 3600, WhPerG: s.Wel / 3600 / sc.alMass,
    ventL: (s.vent + s.purge) * VM, ventPct: s.gen > 0 ? 100 * (s.vent + s.purge) / s.gen : 0, minSF: pk.minSF, starveMin: s.tStarve / 60,
    peakRoom: pk.room, alLeft: 100 * s.nAl / s.nAl0, dosed: s.dosed, evap: s.evap, ...ev,
    E: { chem: reacted * (-p.dH + 1.5 * LHV) / 1e3, heat: s.Qrx / 1e3, loss: s.Qloss / 1e3, h2: s.gen * LHV / 1e3, stackIn: s.stack * LHV / 1e3,
      elec: s.Wel / 1e3, stackHeat: (s.stack * LHV - s.Wel) / 1e3, vented: (s.vent + s.purge) * LHV / 1e3, head: s.nH2 * LHV / 1e3 },
  };
  return { o, sum, s };
}

export function q(sorted, f) { const n = sorted.length; if (!n) return NaN; const x = f * (n - 1), i = Math.floor(x), d = x - i; return i + 1 < n ? sorted[i] * (1 - d) + sorted[i + 1] * d : sorted[i]; }
const SUMK = ['total', 'peakFlow', 'peakT', 'peakP', 't90', 'Wh', 'WhPerG', 'ventPct', 'minSF', 'peakRoom', 'starveMin', 'pct'];
const EVK = ['tVent', 'tBoil', 'tHdpe', 'tRoom', 'tSF'];
export function ensemble(sc, base, nm, seed, opt = {}, keys) {
  const rng = mulberry32(seed); const runs = [];
  for (let m = 0; m < nm; m++) runs.push(simulate(sc, drawParams(base, rng, keys), opt));
  const band = {}, n = runs[0].o.t.length;
  for (const k of ['h2', 'h2act', 'flow', 'T', 'P', 'Ps', 'Vs', 'room', 'c', 'sf', 'psi', 'Da']) {
    const lo = [], hi = [], md = [], sd = [];
    for (let i = 0; i < n; i++) {
      const col = runs.map(r => r.o[k][i]).sort((a, b) => a - b);
      lo.push(q(col, 0.05)); hi.push(q(col, 0.95)); md.push(q(col, 0.5));
      const mu = col.reduce((a, b) => a + b, 0) / col.length; sd.push(Math.sqrt(col.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(col.length - 1, 1)));
    }
    band[k] = { lo, hi, md, sd };
  }
  const sum = {};
  for (const k of SUMK) { const v = runs.map(r => r.sum[k]).filter(x => x != null).sort((a, b) => a - b); sum[k] = { lo: q(v, 0.05), md: q(v, 0.5), hi: q(v, 0.95), n: v.length }; }
  for (const k of EVK) { const v = runs.map(r => r.sum[k]).filter(x => x != null).sort((a, b) => a - b); sum[k] = { frac: v.length / nm, lo: q(v, 0.05), hi: q(v, 0.95) }; }
  return { band, sum, t: runs[0].o.t };
}

export function polarBand(base, nm, seed, pH2 = 1.5) {
  const rng = mulberry32(seed + 1), I = []; for (let i = 0; i <= 50; i++) I.push(i * 0.01);
  const ps = Array.from({ length: nm }, () => drawParams(base, rng, ['i0a', 'i0c', 'asr']));
  const lo = [], hi = [], md = [], nom = [];
  for (const i of I) { const v = ps.map(p => polar(i, p, pH2).V).sort((a, b) => a - b); lo.push(q(v, .05)); hi.push(q(v, .95)); md.push(q(v, .5)); nom.push(polar(i, base, pH2).V); }
  return { i: I, lo, hi, md, nom, parts: polar(0.2, base, pH2) };
}


// Code-verification fixture. Used only by the Docs verification checks; never shown as a scenario or as data.
export const VERIFY_SC = { alMass: 2, form: 'foil', dim: 20, alloy: 'pure', cNaOH: 2, vLiq: 200, T0: 25, vVessel: 500, mode: 'open', pRelief: 0, cells: 0, cellArea: 1, load: 'none', duration: 1800, tAmb: 25, roomVol: 1e9, ach: 0 };

export function interp(xs, ys, x) {
  if (x <= xs[0]) return ys[0]; const n = xs.length; if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
  const f = (x - xs[lo]) / (xs[hi] - xs[lo]); return ys[lo] * (1 - f) + ys[hi] * f;
}

// ---------- measured datasets ----------
export const METHODS = { wd: 'Water displacement (wet gas)', syr: 'Gas syringe (dry)', flow: 'Dry flowmeter / totaliser', mass: 'Mass loss (entered as mL STP)', stp: 'Already STP (no correction)' };
// Volumetric corrections: wet-gas vapour pressure (water displacement only), then T_amb, P_amb → STP (0 °C, 1 atm).
// Not corrected: hydrostatic head, H2 dissolved in collection water, headspace air expansion.
export function toStp(d) {
  if (d.method === 'mass' || d.method === 'stp') return d.v.slice();
  const Ta = d.cond.tAmb + 273.15, Pa = d.cond.pAmb * 1000, pw = d.method === 'wd' ? psat(Ta) : 0;
  return d.v.map(v => v * (Pa - pw) / P_ATM * 273.15 / Ta);
}
export function condToSc(c, duration) {
  return { alMass: c.alMass, form: c.form, dim: c.dim, alloy: c.alloy, cNaOH: c.cNaOH, vLiq: c.vLiq, T0: c.T0, vVessel: c.vVessel, mode: 'open', pRelief: 0, cells: 0, cellArea: 1, load: 'none', duration, tAmb: c.tAmb, roomVol: 1e9, ach: 0 };
}
export function prep(d) {
  const tmax = d.t[d.t.length - 1];
  return { id: d.id, name: d.name, sc: condToSc(d.cond, Math.ceil(tmax / 3) * 3 + 3), t: d.t, v: toStp(d), T: d.T && d.T.length ? d.T : null, sv: d.cond.sV, sT: d.cond.sT || null, tauT: d.cond.tauT || 0 };
}
function simAt(sc, p) { const r = simulate(sc, p, { dt: 3, every: 1 }); return { ts: r.o.t.map(x => x * 60), v: r.o.h2.map(x => x * 1000), T: r.o.T }; }
function chi2run(p, run) {
  const m = simAt(run.sc, p); let s = 0;
  for (let i = 0; i < run.t.length; i++) {
    s += ((run.v[i] - interp(m.ts, m.v, run.t[i])) / run.sv) ** 2;
    if (run.T && run.sT) s += ((run.T[i] - interp(m.ts, m.T, run.t[i])) / run.sT) ** 2;
  }
  return s;
}
const chi2all = (p, runs) => runs.reduce((a, r) => a + chi2run(p, r), 0);
export function nelderMead(f, x0, st, it = 120) {
  const n = x0.length; let S = [x0.slice()];
  for (let i = 0; i < n; i++) { const x = x0.slice(); x[i] += st[i]; S.push(x); }
  let V = S.map(f);
  for (let k = 0; k < it; k++) {
    const id = V.map((v, i) => i).sort((a, b) => V[a] - V[b]); S = id.map(i => S[i]); V = id.map(i => V[i]);
    const c = Array(n).fill(0); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += S[i][j] / n;
    const w = S[n], xr = c.map((ci, j) => 2 * ci - w[j]), fr = f(xr);
    if (fr < V[0]) { const xe = c.map((ci, j) => 3 * ci - 2 * w[j]), fe = f(xe); if (fe < fr) { S[n] = xe; V[n] = fe; } else { S[n] = xr; V[n] = fr; } }
    else if (fr < V[n - 1]) { S[n] = xr; V[n] = fr; }
    else { const xc = c.map((ci, j) => 0.5 * (ci + w[j])), fc = f(xc); if (fc < V[n]) { S[n] = xc; V[n] = fc; } else { for (let i = 1; i <= n; i++) { S[i] = S[i].map((v, j) => S[0][j] + 0.5 * (v - S[0][j])); V[i] = f(S[i]); } } }
  }
  let b = 0; for (let i = 1; i <= n; i++) if (V[i] < V[b]) b = i; return { x: S[b], f: V[b] };
}

// Calibrate k_s,25, E_a, τ_ind on measured runs (calSets), score held-out runs (valSets). Datasets are raw uploads; prep() applies STP corrections.
export function calibrate(base, calSets, valSets) {
  const runs = calSets.map(prep), vruns = valSets.map(prep);
  const mk = x => ({ ...base, ks25: Math.exp(x[0]), Ea: x[1] * 1e4, tauInd: Math.exp(x[2]) });
  const A = nelderMead(x => chi2all(mk(x), runs), [Math.log(base.ks25), base.Ea / 1e4, Math.log(base.tauInd)], [0.5, 0.5, 0.5], 140);
  const mkB = x => ({ ...base, ks25: Math.exp(x[0]), Ea: x[1] * 1e4, tauInd: 1e-3 });
  const B = nelderMead(x => chi2all(mkB(x), runs), [A.x[0], A.x[1]], [0.4, 0.4], 90);
  const N = runs.reduce((a, r) => a + r.t.length * (r.T && r.sT ? 2 : 1), 0);
  const models = [
    { name: 'L1 + oxide induction', k: 3, chi2: A.f, aic: A.f + 6, bic: A.f + 3 * Math.log(N) },
    { name: 'L1, no induction', k: 2, chi2: B.f, aic: B.f + 4, bic: B.f + 2 * Math.log(N) },
  ];
  const g2 = (a, b) => chi2all(mk([a, b, A.x[2]]), runs) / 2;
  const x0 = A.x, h = 0.01, f0 = g2(x0[0], x0[1]);
  const Haa = (g2(x0[0] + h, x0[1]) - 2 * f0 + g2(x0[0] - h, x0[1])) / h ** 2;
  const Hbb = (g2(x0[0], x0[1] + h) - 2 * f0 + g2(x0[0], x0[1] - h)) / h ** 2;
  const Hab = (g2(x0[0] + h, x0[1] + h) - g2(x0[0] + h, x0[1] - h) - g2(x0[0] - h, x0[1] + h) + g2(x0[0] - h, x0[1] - h)) / (4 * h * h);
  const det = Haa * Hbb - Hab * Hab;
  let sa = 0.3, sb = 0.8; if (det > 0 && Haa > 0) { sa = Math.min(Math.sqrt(Hbb / det), 2); sb = Math.min(Math.sqrt(Haa / det), 3); }
  const G = 15, ga = [], gb = []; for (let i = 0; i < G; i++) { ga.push(x0[0] + (i / (G - 1) - 0.5) * 8 * sa); gb.push(x0[1] + (i / (G - 1) - 0.5) * 8 * sb); }
  const L = []; let mn = Infinity;
  for (let j = 0; j < G; j++) { const row = []; for (let i = 0; i < G; i++) { const v = g2(ga[i], gb[j]); row.push(v); if (v < mn) mn = v; } L.push(row); }
  let W = 0, ma = 0, mb = 0; const w = L.map(r => r.map(v => { const e = Math.exp(-(v - mn)); W += e; return e; }));
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { ma += w[j][i] * ga[i] / W; mb += w[j][i] * gb[j] / W; }
  let va = 0, vb = 0, cab = 0;
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const ww = w[j][i] / W; va += ww * (ga[i] - ma) ** 2; vb += ww * (gb[j] - mb) ** 2; cab += ww * (ga[i] - ma) * (gb[j] - mb); }
  const rng = mulberry32(99), draws = [], cells = [];
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) cells.push([i, j, w[j][i] / W]);
  for (let m = 0; m < 40; m++) { const u = rng(); let acc = 0, c = cells[cells.length - 1]; for (const cc of cells) { acc += cc[2]; if (acc >= u) { c = cc; break; } }
    draws.push(mk([ga[c[0]] + (rng() - 0.5) * (ga[1] - ga[0]), gb[c[1]] + (rng() - 0.5) * (gb[1] - gb[0]), A.x[2]])); }
  const pMap = mk(A.x);
  const pred = run => {
    const sims = draws.map(p => simAt(run.sc, p)), fit = simAt(run.sc, pMap);
    const o = { lo90: [], hi90: [], lo80: [], hi80: [], lo50: [], hi50: [], md: [], Tlo: [], Thi: [], fitV: [], fitT: [] };
    for (let i = 0; i < run.t.length; i++) {
      const col = sims.map(s => interp(s.ts, s.v, run.t[i]) + run.sv * randn(rng)).sort((a, b) => a - b);
      o.lo90.push(q(col, .05)); o.hi90.push(q(col, .95)); o.lo80.push(q(col, .1)); o.hi80.push(q(col, .9)); o.lo50.push(q(col, .25)); o.hi50.push(q(col, .75)); o.md.push(q(col, .5));
      o.fitV.push(interp(fit.ts, fit.v, run.t[i])); o.fitT.push(interp(fit.ts, fit.T, run.t[i]));
      if (run.T) { const ct = sims.map(s => interp(s.ts, s.T, run.t[i]) + (run.sT || 0) * randn(rng)).sort((a, b) => a - b); o.Tlo.push(q(ct, .05)); o.Thi.push(q(ct, .95)); }
    }
    let se = 0, mx = 0, c50 = 0, c80 = 0, c90 = 0, seT = 0;
    for (let i = 0; i < run.t.length; i++) { const e = run.v[i] - o.md[i]; se += e * e; mx = Math.max(mx, Math.abs(e));
      if (run.T) seT += (run.T[i] - o.fitT[i]) ** 2;
      if (run.v[i] >= o.lo50[i] && run.v[i] <= o.hi50[i]) c50++; if (run.v[i] >= o.lo80[i] && run.v[i] <= o.hi80[i]) c80++; if (run.v[i] >= o.lo90[i] && run.v[i] <= o.hi90[i]) c90++; }
    const n = run.t.length;
    return { ...run, ...o, resid: run.v.map((v, i) => v - o.fitV[i]), rmse: Math.sqrt(se / n), maxErr: mx, rmseT: run.T ? Math.sqrt(seT / n) : null,
      peakTErr: run.T ? Math.max(...o.fitT) - Math.max(...run.T) : null, c50: c50 / n, c80: c80 / n, c90: c90 / n };
  };
  return { pMap, x: A.x, models, n: N,
    grid: { ga, gb, w: w.map(r => r.map(v => v / W)), mx: Math.max(...w.flat()) / W },
    post: { ks: { m: Math.exp(ma), lo: Math.exp(ma - 1.645 * Math.sqrt(va)), hi: Math.exp(ma + 1.645 * Math.sqrt(va)) }, Ea: { m: mb * 1e4, lo: (mb - 1.645 * Math.sqrt(vb)) * 1e4, hi: (mb + 1.645 * Math.sqrt(vb)) * 1e4 }, corr: cab / Math.sqrt(va * vb) },
    cal: runs.map(pred), val: vruns.map(pred) };
}

// Code verification on synthetic data (not an experiment): generate noisy runs from known parameters, fit, check recovery.
export function recoveryTest() {
  const truth = { ...defaults(), ks25: defaults().ks25 * 1.5, Ea: 50e3, tauInd: 120 }, rng = mulberry32(7);
  const mkSet = (T0, id) => { const sc = { ...VERIFY_SC, T0 }; const r = simulate(sc, truth, { dt: 3, every: 10 });
    return { id, name: 'verification-' + T0, method: 'stp', cond: { ...sc, pAmb: 101.325, sV: 8, sT: 0.3 }, t: r.o.t.map(x => x * 60), v: r.o.h2.map(x => x * 1000 + 8 * randn(rng)), T: r.o.T.map(x => x + 0.3 * randn(rng)) }; };
  const res = calibrate(defaults(), [mkSet(25, 'a'), mkSet(40, 'b')], []);
  const okK = truth.ks25 >= res.post.ks.lo && truth.ks25 <= res.post.ks.hi, okE = truth.Ea >= res.post.Ea.lo && truth.Ea <= res.post.Ea.hi;
  return { pass: okK && okE, truth, post: res.post };
}

// ---------- OED ----------
export function oedRank(base, cands, nm, seed, sV, sT) {
  const out = [];
  for (const sc of cands) {
    const e = ensemble(sc, base, nm, seed, { dt: 4, every: 15 }, ['ks25', 'Ea', 'tauInd', 'n']);
    let iv = 0, iT = 0; for (let i = 0; i < e.t.length; i++) { iv += 0.5 * Math.log(1 + (e.band.h2.sd[i] * 1000) ** 2 / (sV * sV)); if (sT) iT += 0.5 * Math.log(1 + e.band.T.sd[i] ** 2 / (sT * sT)); }
    out.push({ c: sc, sc, iv, iT, score: iv + iT, e });
  }
  return out.sort((a, b) => b.score - a.score);
}

// ---------- optimisation (random search, nominal parameters, user's scenario as template) ----------
export function optimize(base, goal, n, seed, sc0) {
  const rng = mulberry32(seed), pts = [], dt = 3, every = 4;
  for (let k = 0; k < n; k++) {
    const vMax = Math.min(goal.vMax, sc0.vVessel - 20);
    const sc = { ...sc0, cells: 0, load: 'none', alMass: goal.alMin + (goal.alMax - goal.alMin) * rng(), cNaOH: 0.25 + Math.max(goal.maxM - 0.25, 0) * rng(), vLiq: goal.vMin + Math.max(vMax - goal.vMin, 0) * rng() };
    const r = simulate(sc, base, { dt, every });
    let above = 0; for (const fl of r.o.flow) if (fl >= goal.flow) above += dt * every / 60;
    const cost = sc.alMass / 1000 * goal.pAl + sc.cNaOH * sc.vLiq / 1000 * MW.NaOH / 1000 * goal.pNaOH;
    const ok = above >= goal.dur && r.sum.peakT < goal.Tmax && r.sum.ventPct < goal.maxVent;
    pts.push({ sc, dur: above, peakT: r.sum.peakT, vent: r.sum.ventPct, cost, ok, total: r.sum.total });
  }
  pts.forEach(p => { p.pareto = !pts.some(q2 => q2 !== p && q2.dur >= p.dur && q2.peakT <= p.peakT && (q2.dur > p.dur || q2.peakT < p.peakT)); });
  return pts;
}

// ---------- control (simulation only). cfg = { sc, cDose [M], umax [mL/s], cOpen [M], kMis, uaMis } all user-entered ----------
export function simControl(base, kind, gains, cfg) {
  const sc = cfg.sc, plant = { ...base, ks25: base.ks25 * cfg.kMis, UA: base.UA * cfg.uaMis }, umax = cfg.umax;
  const sp = s => loadI(sc, s.t) * sc.cells / (2 * F) * VM * 6e4 * 1.15;
  const flow = s => s.r * 1.5 * VM * 6e4;
  const st = init(sc, plant); if (kind === 'open') st.nOH = Math.max(cfg.cOpen, sc.cNaOH) * sc.vLiq / 1000;
  let I = 0, eP = 0, u = 0, lastC = -1e9;
  const ctrl = s => {
    if (kind === 'open') return 0;
    if (kind === 'pid') {
      if (s.t - lastC >= 4) { const e = sp(s) - flow(s), de = (e - eP) / 4; eP = e;
        const raw = gains.kp * e + gains.ki * I + gains.kd * de; u = Math.min(Math.max(raw, 0), umax); if (raw === u) I += e * 4; lastC = s.t; }
      return u;
    }
    if (s.t - lastC >= 10) {
      let best = 0, bc = Infinity;
      for (const lv of [0, 0.1, 0.25, 0.5, 0.75, 1]) {
        const c = { ...s }; let cost = 0;
        for (let k = 0; k < 40; k++) { step(c, sc, base, 3, lv * umax, cfg.cDose); const e = (flow(c) - sp(c)) / 100; cost += e * e + 4 * Math.max(0, c.T - 343.15) ** 2 / 100; }
        cost += 0.3 * lv; if (cost < bc) { bc = cost; best = lv * umax; }
      }
      u = best; lastC = s.t;
    }
    return u;
  };
  const r = simulate(sc, plant, { dt: 2, every: 5, ctrl, state: st, cDose: cfg.cDose });
  let iae = 0, below = 0; for (let i = 1; i < r.o.t.length; i++) { const s0 = r.o.dem[i] * 1.15; iae += Math.abs(r.o.flow[i] - s0) * 10 / 60; if (s0 > 0 && r.o.flow[i] < 0.9 * s0) below += 10 / 60; }
  return { o: r.o, sp: r.o.dem.map(x => x * 1.15), sum: r.sum, iae, below, dosed: r.sum.dosed };
}
export function autoTune(base, cfg) {
  const sc = cfg.sc, s = init(sc, base); const fl = () => s.r * 1.5 * VM * 6e4;
  for (let i = 0; i < 150; i++) step(s, sc, base, 2, 0, cfg.cDose);
  const f0 = fl(), du = cfg.umax / 2; let t63 = null; const tr = [];
  for (let i = 0; i < 300; i++) { step(s, sc, base, 2, du, cfg.cDose); tr.push(fl()); }
  const f1 = tr[tr.length - 1], K = (f1 - f0) / du; for (let i = 0; i < tr.length; i++) if (t63 == null && tr[i] - f0 >= 0.632 * (f1 - f0)) t63 = (i + 1) * 2;
  const tau = t63 || 600, kp = 0.5 / Math.max(K, 1e-9);
  return { kp, ki: kp / tau, kd: 0, K, tau };
}

// ---------- live twin (EnKF) ----------
// src.kind === 'data': replay a measured dataset (prep()'d). src.kind === 'simtest': filter test against a simulated plant (NOT real data).
export function twinCreate(base, nm, seed, sc, src) {
  const rng = mulberry32(seed), lag = src.kind === 'data' ? src.run.tauT : 15;
  const members = Array.from({ length: nm }, () => { const p = drawParams(base, rng, ['ks25', 'tauInd', 'UA']); const s = init(sc, p); s.Tm = s.T; return { p, s }; });
  const tw = { rng, sc, src, lag, members, hist: [], fc: null, ks0: base.ks25, done: false, hasT: src.kind === 'simtest' || !!(src.run && src.run.T && src.run.sT) };
  if (src.kind === 'simtest') { const pT = { ...base, ks25: base.ks25 * 1.6, tauInd: base.tauInd * 2 }; const ts = init(sc, pT); ts.Tm = ts.T; tw.truth = { p: pT, s: ts, drift: 0 }; }
  return tw;
}
function solve(Pyy) { const m = Pyy.length; if (m === 1) return [[1 / Pyy[0][0]]]; const d = Pyy[0][0] * Pyy[1][1] - Pyy[0][1] * Pyy[1][0]; return [[Pyy[1][1] / d, -Pyy[0][1] / d], [-Pyy[1][0] / d, Pyy[0][0] / d]]; }
export function twinTick(tw, adv = 20) {
  const dt = 2, sc = tw.sc, nst = Math.round(adv / dt), lag = tw.lag;
  const tNow = tw.members[0].s.t + adv;
  const tEnd = tw.src.kind === 'data' ? tw.src.run.t[tw.src.run.t.length - 1] : sc.duration;
  if (tNow > tEnd) { tw.done = true; return tw; }
  for (let k = 0; k < nst; k++) {
    if (tw.truth) { const T = tw.truth.s; step(T, sc, tw.truth.p, dt); T.Tm += lag ? (T.T - T.Tm) * dt / lag : T.T - T.Tm; tw.truth.drift += 0.01 * dt; }
    for (const m of tw.members) { step(m.s, sc, m.p, dt); m.s.Tm += lag ? (m.s.T - m.s.Tm) * dt / lag : m.s.T - m.s.Tm; }
  }
  const rng = tw.rng, t = tw.members[0].s.t;
  let y, sd;
  if (tw.truth) { sd = [4, 0.3]; y = [tw.truth.s.gen * VM * 1000 + tw.truth.drift + 4 * randn(rng), tw.truth.s.Tm - 273.15 + 0.3 * randn(rng)]; }
  else { const r = tw.src.run; sd = [r.sv]; y = [interp(r.t, r.v, t)]; if (tw.hasT) { sd.push(r.sT); y.push(interp(r.t, r.T, t)); } }
  const mo = y.length, M = tw.members, N = M.length;
  const X = M.map(m => [m.s.nAl, m.s.T, m.s.gen, Math.log(m.p.ks25), m.s.Tm]);
  const Y = M.map(m => [m.s.gen * VM * 1000, m.s.Tm - 273.15].slice(0, mo));
  const mx = [0, 0, 0, 0, 0], my = Array(mo).fill(0);
  X.forEach(x => x.forEach((v, i) => mx[i] += v / N)); Y.forEach(v => v.forEach((w, i) => my[i] += w / N));
  const Pyy = Array.from({ length: mo }, (_, a) => Array.from({ length: mo }, (_, b) => a === b ? sd[a] * sd[a] : 0)), Pxy = Array.from({ length: 5 }, () => Array(mo).fill(0));
  for (let n = 0; n < N; n++) { const dy = Y[n].map((v, i) => v - my[i]);
    for (let a = 0; a < mo; a++) for (let b = 0; b < mo; b++) Pyy[a][b] += dy[a] * dy[b] / (N - 1);
    for (let i = 0; i < 5; i++) for (let b = 0; b < mo; b++) Pxy[i][b] += (X[n][i] - mx[i]) * dy[b] / (N - 1); }
  const inv = solve(Pyy), K = Pxy.map(r => inv[0].map((_, j) => r.reduce((a, v, k2) => a + v * inv[k2][j], 0)));
  M.forEach((m, n) => {
    const innov = y.map((v, i) => v + sd[i] * randn(rng) - Y[n][i]);
    const x = X[n].map((v, i) => v + K[i].reduce((a, kk, j) => a + kk * innov[j], 0));
    m.s.nAl = Math.min(Math.max(x[0], 0), m.s.nAl0); m.s.T = x[1]; m.s.gen = Math.max(x[2], 0); m.p.ks25 = Math.exp(x[3]); m.s.Tm = x[4];
    m.s.nOH = Math.max(m.s.nOH + (m.s.nAlu - (m.s.nAl0 - m.s.nAl)), 0); m.s.nAlu = m.s.nAl0 - m.s.nAl;
  });
  const st = arr => { const mu = arr.reduce((a, b) => a + b, 0) / arr.length; return { m: mu, sd: Math.sqrt(arr.reduce((a, b) => a + (b - mu) ** 2, 0) / (arr.length - 1)) }; };
  const rec = { t: t / 60, yV: y[0], yT: mo > 1 ? y[1] : null, ks: st(M.map(m => m.p.ks25 / tw.ks0)), al: st(M.map(m => 100 * m.s.nAl / m.s.nAl0)), ev: st(M.map(m => m.s.gen * VM * 1000)), eT: st(M.map(m => m.s.T - 273.15)) };
  if (tw.truth) Object.assign(rec, { tV: tw.truth.s.gen * VM * 1000, tT: tw.truth.s.T - 273.15, ksTrue: tw.truth.p.ks25 / tw.ks0, alTrue: 100 * tw.truth.s.nAl / tw.truth.s.nAl0 });
  tw.hist.push(rec);
  const remain = sc.duration - t;
  if (remain > 0) {
    const fr = M.map(m => { const c = { ...m.s }; const o = { t: [], v: [], T: [] }; const ns = Math.round(remain / 4);
      for (let k = 0; k <= ns; k++) { if (k % 15 === 0 || k === ns) { o.t.push(c.t / 60); o.v.push(c.gen * VM * 1000); o.T.push(c.T - 273.15); } if (k < ns) step(c, sc, m.p, 4); } return o; });
    const n = fr[0].t.length, b = { t: fr[0].t, vlo: [], vhi: [], Tlo: [], Thi: [] };
    for (let i = 0; i < n; i++) { const cv = fr.map(o => o.v[i]).sort((a, c) => a - c), cT = fr.map(o => o.T[i]).sort((a, c) => a - c);
      b.vlo.push(q(cv, .05)); b.vhi.push(q(cv, .95)); b.Tlo.push(q(cT, .05)); b.Thi.push(q(cT, .95)); }
    const pk = fr.map(o => Math.max(...o.T)).sort((a, c) => a - c); b.peakT = { lo: q(pk, .05), hi: q(pk, .95) };
    const fin = fr.map(o => o.v[o.v.length - 1]).sort((a, c) => a - c); b.fin = { lo: q(fin, .05), hi: q(fin, .95) };
    tw.fc = b;
  }
  return tw;
}

export function hash(obj) { const s = JSON.stringify(obj); let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); } return (h >>> 0).toString(16).padStart(8, '0'); }
