'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type Point = { x: number; y: number };
type SoundZone = { id: string; part: string; name: string; color: string; x: number; y: number; radius: number; file: string };
type AudioRig = { context: AudioContext; node: AudioWorkletNode; gain: GainNode; fxWet: GainNode };

const zones: SoundZone[] = [
  { id: 'miao', part: 'bocca', name: 'Miao', color: '#ff6b4a', x: 47, y: 40, radius: 11, file: '/audio/meow.ogg' },
  { id: 'fusa', part: 'petto', name: 'Fusa', color: '#f4ba37', x: 52, y: 61, radius: 16, file: '/audio/purr.ogg' },
  { id: 'graffio', part: 'divano', name: 'Graffio', color: '#54c7b7', x: 76, y: 72, radius: 22, file: '/audio/scratch.wav' },
  { id: 'scoreggina', part: 'coda', name: 'Scoreggina', color: '#9776f2', x: 27, y: 78, radius: 24, file: '/audio/fart.wav' },
];

// Short STFT/phase-vocoder one-shot engine. The source is read forward once;
// release() trims the current phrase when the pointer leaves the photo.
const phaseVocoderWorklet = `
class PhaseMorphProcessor extends AudioWorkletProcessor {
  constructor() {
    super(); this.sources = []; this.weights = [1, 0, 0, 0]; this.N = 1024; this.H = 256; this.sampleRateValue = sampleRate;
    this.voices = []; this.ring = new Float32Array(16384); this.read = 0; this.available = 0; this.framesUntilNext = 0;
    this.window = new Float32Array(this.N);
    for (let i = 0; i < this.N; i++) this.window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / this.N);
    this.port.onmessage = ({ data }) => { if (data.type === 'sources') this.sources = data.sources; if (data.type === 'trigger') { this.weights = data.weights; this.trigger(); } if (data.type === 'release') this.release(); };
  }
  trigger() { this.ring.fill(0); this.read = 0; this.available = 0; this.framesUntilNext = 0; this.voices = [{ position: 0, remaining: Math.floor(this.sampleRateValue * 1.38), releaseRemaining: null, gain: 0.95 }]; }
  release() { for (const voice of this.voices) voice.releaseRemaining = Math.min(voice.releaseRemaining ?? Infinity, Math.floor(this.sampleRateValue * 0.06)); }
  fft(real, imag, inverse) {
    const n = real.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; } }
    for (let len = 2; len <= n; len <<= 1) { const ang = (inverse ? 2 : -2) * Math.PI / len; const wr = Math.cos(ang), wi = Math.sin(ang); for (let i = 0; i < n; i += len) { let ur = 1, ui = 0; for (let j = 0; j < len / 2; j++) { const p = i + j, q = p + len / 2; const vr = real[q] * ur - imag[q] * ui, vi = real[q] * ui + imag[q] * ur; real[q] = real[p] - vr; imag[q] = imag[p] - vi; real[p] += vr; imag[p] += vi; const next = ur * wr - ui * wi; ui = ur * wi + ui * wr; ur = next; } } }
    if (inverse) for (let i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
  }
  makeFrame() {
    if (!this.sources.length) return;
    if (!this.voices.length) { this.available += this.H; return; }
    const start = (this.read + this.available) % this.ring.length;
    for (const voice of this.voices) {
      const reals = [], imags = [];
      for (const source of this.sources) { const real = new Float32Array(this.N), imag = new Float32Array(this.N); for (let i = 0; i < this.N; i++) { const index = Math.floor(voice.position + i); real[i] = index < source.length ? source[index] * this.window[i] : 0; } this.fft(real, imag, false); reals.push(real); imags.push(imag); }
      const outReal = new Float32Array(this.N), outImag = new Float32Array(this.N);
      for (let k = 0; k < this.N; k++) { let magnitude = 0, phaseX = 0, phaseY = 0, total = 0; for (let s = 0; s < reals.length; s++) { const weight = this.weights[s] || 0, magnitudeS = Math.hypot(reals[s][k], imags[s][k]), phase = Math.atan2(imags[s][k], reals[s][k]); magnitude += weight * magnitudeS; phaseX += weight * Math.cos(phase); phaseY += weight * Math.sin(phase); total += weight; } const phase = Math.atan2(phaseY, phaseX); outReal[k] = total ? magnitude * Math.cos(phase) : 0; outImag[k] = total ? magnitude * Math.sin(phase) : 0; }
      this.fft(outReal, outImag, true);
      const attack = Math.min(1, voice.position / (this.sampleRateValue * 0.018)); const release = voice.releaseRemaining === null ? 1 : Math.min(1, voice.releaseRemaining / (this.sampleRateValue * 0.06));
      for (let i = 0; i < this.N; i++) this.ring[(start + i) % this.ring.length] += outReal[i] * this.window[i] * attack * release * voice.gain * 1.75;
      voice.position += this.H; voice.remaining -= this.H; if (voice.releaseRemaining !== null) voice.releaseRemaining -= this.H;
    }
    this.voices = this.voices.filter((voice) => voice.remaining > 0 && (voice.releaseRemaining === null || voice.releaseRemaining > 0)); this.available += this.H;
  }
  process(_, outputs) { const output = outputs[0][0]; for (let i = 0; i < output.length; i++) { if (this.framesUntilNext === 0) { this.makeFrame(); this.framesUntilNext = this.H; } output[i] = this.available ? Math.max(-1, Math.min(1, this.ring[this.read])) : 0; this.ring[this.read] = 0; this.read = (this.read + 1) % this.ring.length; if (this.available) this.available--; this.framesUntilNext--; } return true; }
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

function makeImpulse(context: AudioContext) {
  const length = Math.floor(context.sampleRate * 1.2); const impulse = context.createBuffer(2, length, context.sampleRate);
  for (let channel = 0; channel < 2; channel++) { const data = impulse.getChannelData(channel); for (let i = 0; i < length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / length) ** 2.4; }
  return impulse;
}

function scheduleRhythm(audio: AudioRig, step: number) {
  const { context, gain } = audio; const now = context.currentTime + 0.015; const accented = step % 4 === 0;
  const kick = context.createOscillator(); const kickGain = context.createGain(); kick.type = 'sine'; kick.frequency.setValueAtTime(accented ? 142 : 94, now); kick.frequency.exponentialRampToValueAtTime(48, now + 0.18); kickGain.gain.setValueAtTime(0.0001, now); kickGain.gain.exponentialRampToValueAtTime(accented ? 0.42 : 0.27, now + 0.005); kickGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.2); kick.connect(kickGain).connect(gain); kick.start(now); kick.stop(now + 0.21);
  const hatBuffer = context.createBuffer(1, Math.floor(context.sampleRate * 0.045), context.sampleRate); const noise = hatBuffer.getChannelData(0); for (let i = 0; i < noise.length; i++) noise[i] = Math.random() * 2 - 1;
  const hat = context.createBufferSource(); const filter = context.createBiquadFilter(); const hatGain = context.createGain(); hat.buffer = hatBuffer; filter.type = 'highpass'; filter.frequency.value = 5200; hatGain.gain.setValueAtTime(0.0001, now); hatGain.gain.exponentialRampToValueAtTime(step % 2 ? 0.075 : 0.045, now + 0.002); hatGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.043); hat.connect(filter).connect(hatGain).connect(gain); hat.start(now); hat.stop(now + 0.05);
}

export default function Home() {
  const [active, setActive] = useState<Point>({ x: 47, y: 40 }); const [weights, setWeights] = useState([1, 0, 0, 0]); const [isMuted, setIsMuted] = useState(false);
  const [sequence, setSequence] = useState<Array<number | null>>([0, null, 1, null, 2, null, 3, null]); const [isSequencing, setIsSequencing] = useState(false); const [sequencerStep, setSequencerStep] = useState(-1);
  const [harmonyOn, setHarmonyOn] = useState(false); const [rhythmOn, setRhythmOn] = useState(false); const audioRef = useRef<AudioRig | null>(null); const sequenceRef = useRef(sequence); const rhythmTimerRef = useRef<number | null>(null); const isHoldingRef = useRef(false); const harmonyRef = useRef(false); const muteRef = useRef(false);
  const activeZone = useMemo(() => zones.reduce((best, zone, index) => weights[index] > weights[zones.indexOf(best)] ? zone : best, zones[0]), [weights]);

  const startAudio = useCallback(async () => {
    if (audioRef.current) { await audioRef.current.context.resume(); return audioRef.current; }
    const context = new AudioContext(); const workletUrl = URL.createObjectURL(new Blob([phaseVocoderWorklet], { type: 'application/javascript' })); await context.audioWorklet.addModule(workletUrl); URL.revokeObjectURL(workletUrl);
    const buffers = await Promise.all(zones.map(async (zone) => context.decodeAudioData(await (await fetch(zone.file)).arrayBuffer()))); const node = new AudioWorkletNode(context, 'phase-morph-processor'); const gain = context.createGain(); const fxWet = context.createGain(); const delay = context.createDelay(1); const feedback = context.createGain(); const reverb = context.createConvolver();
    gain.gain.value = muteRef.current ? 0 : 0.72; fxWet.gain.value = harmonyRef.current ? 0.38 : 0; delay.delayTime.value = 0.22; feedback.gain.value = 0.22; reverb.buffer = makeImpulse(context);
    node.connect(gain).connect(context.destination); node.connect(delay); delay.connect(feedback).connect(delay); delay.connect(fxWet); node.connect(reverb).connect(fxWet); fxWet.connect(gain);
    const sources = buffers.map((buffer) => buffer.getChannelData(0).slice()); node.port.postMessage({ type: 'sources', sources }, sources.map((source) => source.buffer)); audioRef.current = { context, node, gain, fxWet }; return audioRef.current;
  }, []);
  const triggerWeights = useCallback(async (nextWeights: number[]) => { const audio = await startAudio(); audio.node.port.postMessage({ type: 'trigger', weights: nextWeights }); }, [startAudio]);
  const releaseSound = useCallback(() => { audioRef.current?.node.port.postMessage({ type: 'release' }); }, []);
  const playPoint = useCallback(async (point: Point) => { const nextWeights = normalizedWeights(point); setActive(point); setWeights(nextWeights); await triggerWeights(nextWeights); }, [triggerWeights]);
  const triggerZone = useCallback(async (zoneIndex: number) => { const zone = zones[zoneIndex]; await playPoint({ x: zone.x, y: zone.y }); }, [playPoint]);
  const toggleSequencer = useCallback(async () => { if (isSequencing) { setIsSequencing(false); setSequencerStep(-1); return; } await startAudio(); setIsSequencing(true); }, [isSequencing, startAudio]);
  const toggleRhythm = useCallback(async () => { if (rhythmOn) { setRhythmOn(false); return; } await startAudio(); setRhythmOn(true); }, [rhythmOn, startAudio]);

  useEffect(() => { sequenceRef.current = sequence; }, [sequence]); useEffect(() => { harmonyRef.current = harmonyOn; }, [harmonyOn]); useEffect(() => { muteRef.current = isMuted; }, [isMuted]);
  useEffect(() => { if (!isSequencing) return undefined; let cursor = 0; const tick = () => { const zoneIndex = sequenceRef.current[cursor]; setSequencerStep(cursor); if (zoneIndex !== null) void triggerZone(zoneIndex); cursor = (cursor + 1) % sequenceRef.current.length; }; tick(); const interval = window.setInterval(tick, 510); return () => window.clearInterval(interval); }, [isSequencing, triggerZone]);
  useEffect(() => { if (!rhythmOn) { if (rhythmTimerRef.current !== null) window.clearInterval(rhythmTimerRef.current); rhythmTimerRef.current = null; return undefined; } let step = 0; const tick = () => { if (audioRef.current) scheduleRhythm(audioRef.current, step); step = (step + 1) % 8; }; tick(); rhythmTimerRef.current = window.setInterval(tick, 510); return () => { if (rhythmTimerRef.current !== null) window.clearInterval(rhythmTimerRef.current); rhythmTimerRef.current = null; }; }, [rhythmOn]);
  useEffect(() => { if (audioRef.current) audioRef.current.gain.gain.value = isMuted ? 0 : 0.72; }, [isMuted]); useEffect(() => { if (audioRef.current) audioRef.current.fxWet.gain.value = harmonyOn ? 0.38 : 0; }, [harmonyOn]); useEffect(() => () => { audioRef.current?.context.close(); }, []);

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => { event.currentTarget.setPointerCapture(event.pointerId); isHoldingRef.current = true; const box = event.currentTarget.getBoundingClientRect(); void playPoint({ x: ((event.clientX - box.left) / box.width) * 100, y: ((event.clientY - box.top) / box.height) * 100 }).then(() => { if (!isHoldingRef.current) releaseSound(); }); };
  const handlePointerUp = () => { isHoldingRef.current = false; releaseSound(); };
  const cycleStep = (index: number) => setSequence((current) => current.map((value, step) => step !== index ? value : value === null ? 0 : value === zones.length - 1 ? null : value + 1));

  return (
    <main className="site-shell">
      <button className="mute-button" onClick={() => setIsMuted((muted) => !muted)} aria-label={isMuted ? 'Attiva audio' : 'Disattiva audio'} aria-pressed={isMuted}>{isMuted ? '◌' : '◉'}</button>
      <section className="photo-stage"><div className="cat-photo-wrap" onPointerDown={handlePointerDown} onPointerUp={handlePointerUp} onPointerCancel={handlePointerUp} onLostPointerCapture={handlePointerUp} role="button" tabIndex={0} aria-label="Tocca la foto del gatto"><img src="/images/cat.jpg" alt="Il gatto del Cattofono" className="cat-photo" draggable={false} /><div className="photo-shade" /><div className="crosshair" style={{ left: `${active.x}%`, top: `${active.y}%` }}><span /><span /></div>{zones.map((zone) => <span key={zone.id} className={`epicenter ${activeZone.id === zone.id ? 'selected' : ''}`} style={{ left: `${zone.x}%`, top: `${zone.y}%`, '--zone-color': zone.color } as React.CSSProperties}><i /></span>)}</div></section>
      <section className="sequencer-section" aria-label="Sequencer"><div className="seq-panel"><div className="seq-grid">{sequence.map((zoneIndex, index) => <button key={index} className={`seq-step ${sequencerStep === index ? 'playing' : ''} ${zoneIndex === null ? 'empty' : ''}`} onClick={() => cycleStep(index)} aria-label={`Step ${index + 1}: ${zoneIndex === null ? 'vuoto' : zones[zoneIndex].name}`} style={zoneIndex === null ? undefined : { '--step-color': zones[zoneIndex].color } as React.CSSProperties}>{zoneIndex !== null && <i />}</button>)}</div><div className="seq-controls"><button className={`icon-button play ${isSequencing ? 'active' : ''}`} onClick={() => void toggleSequencer()} aria-label={isSequencing ? 'Ferma sequencer' : 'Avvia sequencer'}>{isSequencing ? '■' : '▶'}</button><button className={`icon-button ${harmonyOn ? 'active' : ''}`} onClick={() => setHarmonyOn((value) => !value)} aria-label={harmonyOn ? 'Disattiva delay e riverbero' : 'Attiva delay e riverbero'} aria-pressed={harmonyOn}>♫</button><button className={`icon-button ${rhythmOn ? 'active' : ''}`} onClick={() => void toggleRhythm()} aria-label={rhythmOn ? 'Disattiva base ritmica' : 'Attiva base ritmica'} aria-pressed={rhythmOn}>♩</button></div></div></section>
    </main>
  );
}
