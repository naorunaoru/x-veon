const fs = require('node:fs');
const path = require('node:path');
const file = process.env.XV_RELEASE_FILE || path.join(__dirname, 'out', 'release.json');
if (!fs.existsSync(file)) throw new Error(`${file} is missing: run \`npm run dist\`, which builds the bundle before packaging`);
const release = JSON.parse(fs.readFileSync(file, 'utf8'));
const v = release.platform; // Only tagged builds carry the OS version fields.

module.exports = {
  appId: release.appId,
  productName: release.productName,
  extraMetadata: { version: release.version },
  // The Windows installer takes its fourth version part from buildNumber.
  ...(v ? { buildVersion: v.windowsVersion, buildNumber: v.buildNumber } : {}),
  directories: { output: 'dist' },
  files: ['out/**', '!out/renderer/samples/**', '!out/release.json', 'native/xveon-native.*.node'],
  asarUnpack: ['native/*.node'],
  mac: {
    target: [{ target: 'dmg', arch: ['arm64'] }], identity: '-', artifactName: release.installers.mac,
    ...(v ? { bundleShortVersion: v.macShortVersion, bundleVersion: v.macBundleVersion } : {}),
  },
  win: { target: [{ target: 'nsis', arch: ['x64'] }] },
  nsis: { oneClick: false, perMachine: false, artifactName: release.installers.win },
};
