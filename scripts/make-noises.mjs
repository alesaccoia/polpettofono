import { writeFileSync } from 'node:fs';

const sampleRate = 44100;

function writeWav(path, samples) {
  const pcm = Buffer.alloc(samples.length * 2);
  samples.forEach((sample, index) => pcm.writeInt16LE(Math.max(-1, Math.min(1, sample)) * 32767, index * 2));
  const header = Buffer.alloc(44);
  header.write('RIFF', 0); header.writeUInt32LE(36 + pcm.length, 4); header.write('WAVE', 8);
  header.write('fmt ', 12); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24); header.writeUInt32LE(sampleRate * 2, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34);
  header.write('data', 36); header.writeUInt32LE(pcm.length, 40);
  writeFileSync(path, Buffer.concat([header, pcm]));
}

const fartLength = Math.floor(sampleRate * 0.72);
const fart = Array.from({ length: fartLength }, (_, index) => {
  const time = index / sampleRate;
  const envelope = Math.min(1, time / 0.025) * Math.min(1, (0.72 - time) / 0.18);
  const pulse = 0.5 + 0.5 * Math.sin(time * 2 * Math.PI * 7);
  const frequency = 92 + 42 * Math.exp(-time * 2.7) + pulse * 12;
  const phase = 2 * Math.PI * (frequency * time + 42 * time * time);
  const buzz = Math.sin(phase) * 0.52 + Math.sin(phase * 2.03) * 0.23 + Math.sin(phase * 3.07) * 0.1;
  const air = (Math.random() * 2 - 1) * 0.12;
  return (buzz + air) * envelope * 0.78;
});

const scratchLength = Math.floor(sampleRate * 1.08);
const scratchCenters = [0.08, 0.27, 0.46, 0.65, 0.84];
const scratch = Array.from({ length: scratchLength }, (_, index) => {
  const time = index / sampleRate;
  const rasp = scratchCenters.reduce((value, center, scratchIndex) => {
    const local = time - center;
    const burst = Math.exp(-((local / 0.058) ** 2));
    const grit = (Math.random() * 2 - 1) * 0.6 + Math.sin(local * (13000 + scratchIndex * 1100)) * 0.18;
    return value + burst * grit;
  }, 0);
  const body = Math.sin(time * 2 * Math.PI * 180) * 0.055;
  const fade = Math.min(1, time / 0.02) * Math.min(1, (1.08 - time) / 0.12);
  return (rasp + body) * fade * 0.72;
});

writeWav('public/audio/fart.wav', fart);
writeWav('public/audio/scratch.wav', scratch);
