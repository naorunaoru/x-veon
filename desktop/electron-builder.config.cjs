const { execFileSync } = require('node:child_process');

const sha = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();

module.exports = {
  appId: 'io.github.naorunaoru.xveon.beta',
  productName: 'X-veon Beta',
  extraMetadata: { version: `0.0.0-dev.${sha}` },
  directories: { output: 'dist' },
  files: ['out/**', '!out/renderer/samples/**', 'native/xveon-native.*.node'],
  asarUnpack: ['native/*.node'],
  mac: { target: [{ target: 'dmg', arch: ['arm64'] }], identity: '-' },
  win: { target: [{ target: 'nsis', arch: ['x64'] }] },
  nsis: { oneClick: false, perMachine: false },
};
