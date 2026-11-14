// LRC parsing and title clean-up of web/lyrics.js (was tests/test_core.py for the Python server).
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
const window = {};
new Function('window', readFileSync(new URL('../../src/chordstream/web/lyrics.js', import.meta.url), 'utf8'))(window);
const { parseLrc, cleanTitle } = window.lrclib;
const lines = parseLrc('[00:12.50] Hello there\n[01:02.00]\n[00:03.10]First\nnot a timed line');
assert.deepEqual(lines.map((l) => l.t), [3.1, 12.5, 62]);
assert.equal(lines[0].text, 'First');
assert.equal(lines[2].text, '');  // blank = instrumental break
assert.equal(cleanTitle('In the End (Official Audio)'), 'In the End');
assert.equal(cleanTitle('Song [Lyric Video] (Remastered 2009)'), 'Song');
console.log('lyrics ok');
