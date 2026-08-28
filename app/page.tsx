'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type Point = { x: number; y: number };
type SoundZone = {
  id: string;
  name: string;
  sound: string;
  color: string;
  x: number;
  y: number;
  radius: number;
  file: string;
};

const zones: SoundZone[] = [
  { id: 'miao', name: 'Miao', sound: 'miao', color: '#ff6b4a', x: 49, y: 64, radius: 12, file: '/audio/meow.ogg' },
  { id: 'fusa', name: 'Fusa', sound: 'fusa', color: '#f4ba37', x: 50, y: 83, radius: 15, file: '/audio/purr.ogg' },
  { id: 'chiamata', name: 'Chiamata', sound: 'chiamata', color: '#54c7b7', x: 50, y: 46, radius: 15, file: '/audio/cat-call.ogg' },
  { id: 'verso', name: 'Verso strano', sound: 'verso', color: '#9776f2', x: 24, y: 67, radius: 20, file: '/audio/cat-funny.ogg' },
];

const phaseVocoderWorklet = `
class PhaseMorphProcessor extends AudioWorkletProcessor {
  constructor() {
    super(); this.sources = []; this.weights = [1, 0, 0, 0]; this.N = 1024; this.H = 256;
    this.ring = new Float32Array(16384); this.read = 0; this.available = 0; this.framesUntilNext = 0; this.sourcePosition = 0;
    this.window = new Float32Array(this.N);
    for (let i = 0; i < this.N; i++) this.window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / this.N);
    this.port.onmessage = ({ data }) => {
      if (data.type === 'sources') this.sources = data.sources;
      if (data.type === 'weights') this.weights = data.weights;
      if (data.type === 'reset') { this.sourcePosition = 0; this.ring.fill(0); this.read = 0; this.available = 0; this.framesUntilNext = 0; }
    };
  }
  fft(real, imag, inverse) {
    const n = real.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; } }
    for (let len = 2; len <= n; len <<= 1) { const ang = (inverse ? 2 : -2) * Math.PI / len; const wr = Math.cos(ang), wi = Math.sin(ang); for (let i = 0; i < n; i += len) { let ur = 1, ui = 0; for (let j = 0; j < len / 2; j++) { const p = i + j, q = p + len / 2; const vr = real[q] * ur - imag[q] * ui; const vi = real[q] * ui + imag[q] * ur; real[q] = real[p] - vr; imag[q] = imag[p] - vi; real[p] += vr; imag[p] += vi; const next = ur * wr - ui * wi; ui = ur * wi + ui * wr; ur = next; } } }
    if (inverse) for (let i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
  }
  makeFrame() {
    if (!this.sources.length) return;
    const reals = [], imags = [];
    for (const source of this.sources) { const real = new Float32Array(this.N), imag = new Float32Array(this.N); for (let i = 0; i < this.N; i++) real[i] = source[(this.sourcePosition + i) % source.length] * this.window[i]; this.fft(real, imag, false); reals.push(real); imags.push(imag); }
    const outReal = new Float32Array(this.N), outImag = new Float32Array(this.N);
    for (let k = 0; k < this.N; k++) { let magnitude = 0, phaseX = 0, phaseY = 0, total = 0; for (let s = 0; s < reals.length; s++) { const weight = this.weights[s] || 0, magnitudeS = Math.hypot(reals[s][k], imags[s][k]), phase = Math.atan2(imags[s][k], reals[s][k]); magnitude += weight * magnitudeS; phaseX += weight * Math.cos(phase); phaseY += weight * Math.sin(phase); total += weight; } const phase = Math.atan2(phaseY, phaseX); outReal[k] = total ? magnitude * Math.cos(phase) : 0; outImag[k] = total ? magnitude * Math.sin(phase) : 0; }
    this.fft(outReal, outImag, true); const start = (this.read + this.available) % this.ring.length; for (let i = 0; i < this.N; i++) this.ring[(start + i) % this.ring.length] += outReal[i] * this.window[i] * 1.8; this.sourcePosition = (this.sourcePosition + this.H) % Math.max(1, this.sources[0].length); this.available += this.H;
  }
  process(_, outputs) {
    const output = outputs[0][0];
    for (let i = 0; i < output.length; i++) { if (this.framesUntilNext === 0) { this.makeFrame(); this.framesUntilNext = this.H; } output[i] = this.available ? Math.max(-1, Math.min(1, this.ring[this.read])) : 0; this.ring[this.read] = 0; this.read = (this.read + 1) % this.ring.length; if (this.available) this.available--; this.framesUntilNext--; }
    return true;
  }
}
registerProcessor('phase-morph-processor', PhaseMorphProcessor);
`;

function normalizedWeights(point: Point) {
  const raw = zones.map((zone) => Math.max(0, 1 - Math.hypot(point.x - zone.x, (point.y - zone.y) * 1.08) / zone.radius));
  const sum = raw.reduce((total, value) => total + value, 0);
  if (sum > 0) return raw.map((value) => value / sum);
  const inverse = zones.map((zone) => 1 / Math.max(3, Math.hypot(point.x - zone.x, (point.y - zone.y) * 1.08)));
  const inverseSum = inverse.reduce((total, value) => total + value, 0);
  return inverse.map((value) => value / inverseSum);
}

export default function Home() {
  const [active, setActive] = useState<Point>({ x: 49, y: 64 });
  const [weights, setWeights] = useState([1, 0, 0, 0]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [tapCount, setTapCount] = useState(0);
  const [lastLabel, setLastLabel] = useState('tocca per suonare');
  const audioRef = useRef<{ context: AudioContext; node: AudioWorkletNode; gain: GainNode } | null>(null);
  const activeZone = useMemo(() => zones.reduce((best, zone, index) => weights[index] > weights[zones.indexOf(best)] ? zone : best, zones[0]), [weights]);

  const startAudio = useCallback(async () => {
    if (audioRef.current) { await audioRef.current.context.resume(); setIsPlaying(true); return audioRef.current; }
    const context = new AudioContext();
    const workletUrl = URL.createObjectURL(new Blob([phaseVocoderWorklet], { type: 'application/javascript' }));
    await context.audioWorklet.addModule(workletUrl); URL.revokeObjectURL(workletUrl);
    const buffers = await Promise.all(zones.map(async (zone) => context.decodeAudioData(await (await fetch(zone.file)).arrayBuffer())));
    const node = new AudioWorkletNode(context, 'phase-morph-processor'); const gain = context.createGain(); gain.gain.value = 0.72; node.connect(gain).connect(context.destination);
    const sources = buffers.map((buffer) => buffer.getChannelData(0).slice());
    node.port.postMessage({ type: 'sources', sources }, sources.map((source) => source.buffer)); node.port.postMessage({ type: 'weights', weights: [1, 0, 0, 0] });
    audioRef.current = { context, node, gain }; setIsReady(true); setIsPlaying(true); return audioRef.current;
  }, []);

  const playPoint = useCallback(async (point: Point) => {
    const nextWeights = normalizedWeights(point); const nearest = zones.reduce((best, zone, index) => nextWeights[index] > nextWeights[zones.indexOf(best)] ? zone : best, zones[0]);
    setActive(point); setWeights(nextWeights); setTapCount((count) => count + 1); setLastLabel(nearest.name);
    const audio = await startAudio(); audio.node.port.postMessage({ type: 'weights', weights: nextWeights }); audio.node.port.postMessage({ type: 'reset' });
  }, [startAudio]);

  useEffect(() => { if (audioRef.current) audioRef.current.gain.gain.value = isMuted ? 0 : 0.72; }, [isMuted]);
  useEffect(() => () => { audioRef.current?.context.close(); }, []);
  const handlePointer = (event: React.PointerEvent<HTMLDivElement>) => { const box = event.currentTarget.getBoundingClientRect(); playPoint({ x: ((event.clientX - box.left) / box.width) * 100, y: ((event.clientY - box.top) / box.height) * 100 }); };

  return (
    <main className="site-shell">
      <header className="topbar"><a href="#top" className="brand" aria-label="Il Cattofono home"><span className="brand-mark">◉</span><span>cattofono</span></a><div className="topbar-note"><span className="live-dot" /> phase vocoder / web audio</div><button className="sound-toggle" onClick={() => setIsMuted((muted) => !muted)} aria-label={isMuted ? 'Attiva audio' : 'Disattiva audio'}>{isMuted ? 'audio off' : 'audio on'} <span className="toggle-dot" /></button></header>
      <section id="top" className="intro-grid">
        <div className="intro-copy"><div className="eyebrow"><span>strumento felino</span><span className="eyebrow-line" /></div><h1>suona il<br /><em>gatto.</em></h1><p className="intro-description">Un piccolo sintetizzatore a pelo corto. Tocca la foto: ogni zona attiva un verso diverso, i punti intermedi fanno <strong>morphing</strong>.</p><button className="start-button" onClick={() => playPoint(active)}><span>{isPlaying ? 'ri-suona il punto' : 'attiva il cattofono'}</span><span className="button-arrow">↗</span></button><div className="micro-copy"><span className="key-hint">click / tap</span><span>non serve saper suonare</span></div></div>
        <div className="instrument-wrap"><div className="instrument-topline"><span>strumento #001</span><span>{String(tapCount).padStart(2, '0')} tocchi</span></div><div className="cat-card"><div className="cat-photo-wrap" onPointerDown={handlePointer} role="button" tabIndex={0} aria-label="Tocca una parte del gatto per ascoltare il suo suono" onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') playPoint(active); }}><img src="/images/cat.jpg" alt="Gatto rosso che guarda in camera" className="cat-photo" draggable={false} /><div className="photo-shade" /><div className="crosshair" style={{ left: `${active.x}%`, top: `${active.y}%` }}><span /><span /></div>{zones.map((zone) => <span key={zone.id} className={`epicenter ${activeZone.id === zone.id ? 'selected' : ''}`} style={{ left: `${zone.x}%`, top: `${zone.y}%`, '--zone-color': zone.color } as React.CSSProperties}><i /></span>)}<div className="tap-label" style={{ left: `${active.x}%`, top: `${active.y}%` }}><span>{lastLabel}</span><b>+</b></div></div><div className="cat-caption"><div><span className="caption-kicker">ora stai suonando</span><strong>{activeZone.name}</strong></div><div className="waveform" aria-hidden="true"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div><span className="freq-readout">{Math.round(weights[zones.indexOf(activeZone)] * 100)}%</span></div></div><div className="zone-legend">{zones.map((zone, index) => <div className="legend-item" key={zone.id}><span className="legend-swatch" style={{ backgroundColor: zone.color }} /><span>{zone.name}</span><span className="legend-value">{Math.round(weights[index] * 100)}%</span></div>)}</div></div>
      </section>
      <section className="how-it-works"><div className="section-label">01 / come funziona</div><div className="process-row"><div className="process-item"><span className="process-number">01</span><div><h2>trova un epicentro</h2><p>Bocca, petto, orecchie e coda hanno ognuno una firma sonora.</p></div></div><span className="process-arrow">→</span><div className="process-item"><span className="process-number">02</span><div><h2>spostati tra i suoni</h2><p>La distanza dai punti miscela i campioni in tempo reale.</p></div></div><span className="process-arrow">→</span><div className="process-item"><span className="process-number">03</span><div><h2>crea il tuo verso</h2><p>Ogni tocco è diverso. Anche il gatto non sa cosa succederà.</p></div></div></div></section>
      <footer className="footer"><span>fatto con curiosità e baffi</span><span>foto + suoni: Wikimedia Commons · CC / GFDL</span><span>© 2026 cattofono</span></footer>
      {!isReady && <div className="audio-nudge">Premi un punto per iniziare l’audio <span>↗</span></div>}
    </main>
  );
}
