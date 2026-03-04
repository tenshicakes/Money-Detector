const fs = require('fs');
const src = fs.readFileSync('node_modules/inferencejs/dist/inference.es.js', 'utf8');
// Find the base64 blob between quotes before the tv function
const match = src.match(/const cn\s*=\s*"([A-Za-z0-9+/=]+)"/);
if (match) {
  const decoded = Buffer.from(match[1], 'base64').toString('utf8');
  fs.writeFileSync('worker_decoded.js', decoded);
  console.log('Wrote worker_decoded.js, length:', decoded.length);
} else {
  // Try alternate pattern
  const m2 = src.match(/"(KGZ1bmN[A-Za-z0-9+/=]+)"/);
  if (m2) {
    const decoded = Buffer.from(m2[1], 'base64').toString('utf8');
    fs.writeFileSync('worker_decoded.js', decoded);
    console.log('Wrote worker_decoded.js, length:', decoded.length);
  } else {
    console.log('No base64 blob found');
  }
}
