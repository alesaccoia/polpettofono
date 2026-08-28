import { writeFileSync } from 'node:fs';

const sampleRate = 44100;
const duration = 1.12;
const frameCount = Math.floor(sampleRate * duration);
const pcm = Buffer.alloc(frameCount * 2);

for (let i = 0; i < frameCount; i += 1) {
  const time = i / sampleRate;
  let sample = 0;
  for (const center of [0.06, 0.35, 0.67]) {
    const local = time - center;
    const pulse = Math.exp(-((local / 0.105) ** 2));
    const sweep = Math.max(0, Math.min(1, (local + 0.11) / 0.22));
    const phase = 2 * Math.PI * (720 * local + 760 * sweep ** 2 * local);
    sample += pulse * (0.56 * Math.sin(phase) + 0.2 * Math.sin(phase * 2.01) + 0.07 * Math.sin(phase * 3.02));
    sample += pulse * 0.025 * (Math.random() * 2 - 1);
  }
  const fade = Math.min(1, time / 0.025, (duration - time) / 0.1);
  pcm.writeInt16LE(Math.max(-1, Math.min(1, sample * fade * 0.82)) * 32767, i * 2);
}

const header = Buffer.alloc(44);
header.write('RIFF', 0); header.writeUInt32LE(36 + pcm.length, 4); header.write('WAVE', 8);
header.write('fmt ', 12); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
header.writeUInt32LE(sampleRate, 24); header.writeUInt32LE(sampleRate * 2, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34);
header.write('data', 36); header.writeUInt32LE(pcm.length, 40);
writeFileSync('public/audio/chirp.wav', Buffer.concat([header, pcm]));
