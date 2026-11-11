import numpy as np
import pytest

from chordstream.config import FrontendConfig
from chordstream.cqt import StreamingCQT, cqt_frequencies, offline_cqt
from chordstream.data.billboard import salami_to_segments
from chordstream.data.ug import chords_from_store, extract_store, normalize
from chordstream.hmm import OnlineHMM, build_transition
from chordstream.vocab import IGNORE, NO_CHORD, class_index, parse_freetext, parse_harte, transpose

C, Am, G = class_index(0, False), class_index(9, True), class_index(7, False)


@pytest.mark.parametrize("s,want", [
    ("C", C), ("Cmaj7", C), ("C7", C), ("Cadd9", C), ("C/G", C), ("B#", C),
    ("Am", Am), ("Am7", Am), ("Amin", Am), ("A-7", Am), ("Am(maj7)", Am),
    ("Bdim", IGNORE), ("F#m7b5", IGNORE), ("Dsus4", IGNORE), ("E5", IGNORE), ("Caug", IGNORE),
    ("N.C.", NO_CHORD), ("x2", None), ("Hello", None),
])
def test_freetext(s, want):
    assert parse_freetext(s) == want


@pytest.mark.parametrize("s,want", [
    ("C:maj", C), ("A:min7", Am), ("G:7/3", G), ("A:(1,b3,5)", Am), ("N", NO_CHORD),
    ("X", IGNORE), ("D:sus4(b7)", IGNORE), ("B:hdim7", IGNORE), ("C:maj(9)", C),
])
def test_harte(s, want):
    assert parse_harte(s) == want


def test_transpose():
    assert transpose(C, 7) == G
    assert transpose(Am, 3) == class_index(0, True)
    assert transpose(NO_CHORD, 5) == NO_CHORD


def test_ug_store_both_layouts():
    payload = '{"data":{"tab_view":{"wiki_tab":{"content":"[ch]Am[/ch] [ch]Am[/ch]\\n[ch]F[/ch] [ch]Dsus4[/ch] [ch]G7[/ch]"}}}}'
    legacy = f"<script>window.UGAPP.store.page = {payload};</script>"
    import html as h
    current = f'<div class="js-store" data-content="{h.escape(chr(123) + chr(34) + "store" + chr(34) + ":{" + chr(34) + "page" + chr(34) + ":" + payload + "}}")}"></div>'
    for page in (legacy, current):
        raw, _ = chords_from_store(extract_store(page))
        assert raw == ["Am", "Am", "F", "Dsus4", "G7"]
        assert normalize(raw) == [Am, class_index(5, False), IGNORE, G]


def test_cqt_pitch_alignment():
    cfg = FrontendConfig()
    sr = cfg.sample_rate
    t = np.arange(sr) / sr
    a4 = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    feats = offline_cqt(a4, cfg, device="cpu")
    peak = int(feats[-1].argmax())
    assert abs(cqt_frequencies(cfg)[peak] - 440) < 1.0


def test_streaming_matches_offline():
    cfg = FrontendConfig()
    rng = np.random.default_rng(0)
    y = rng.standard_normal(cfg.sample_rate).astype(np.float32) * 0.1
    off = offline_cqt(y, cfg, device="cpu")
    s = StreamingCQT(cfg)
    on = np.concatenate([s.push(y[i : i + 333]) for i in range(0, len(y), 333)])
    assert on.shape == off.shape
    np.testing.assert_allclose(on, off, atol=1e-3)


def test_cqt_is_causal():
    cfg = FrontendConfig()
    rng = np.random.default_rng(1)
    y = rng.standard_normal(cfg.sample_rate).astype(np.float32)
    y2 = y.copy()
    y2[cfg.hop_length * 10 :] = 0  # change the future
    a, b = offline_cqt(y, cfg, device="cpu"), offline_cqt(y2, cfg, device="cpu")
    np.testing.assert_allclose(a[:10], b[:10], atol=1e-4)


def test_model_is_causal():
    torch = pytest.importorskip("torch")
    from chordstream.config import ModelConfig
    from chordstream.model import ChordNet

    net = ChordNet(ModelConfig(d_model=64, n_layers=2, context_frames=32)).eval()
    x = torch.randn(1, 32, 144)
    x2 = x.clone()
    x2[:, 20:] = torch.randn(1, 12, 144)
    with torch.no_grad():
        a, b = net(x), net(x2)
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-5)
    assert not torch.allclose(a[:, 20:], b[:, 20:])


def test_hmm_suppresses_flicker():
    A = build_transition(None, mean_chord_seconds=2.0, frame_seconds=0.04)
    hmm = OnlineHMM(A)
    probs = np.full((30, 25), 0.01)
    probs[:, C] = 0.7
    probs[15, C], probs[15, G] = 0.3, 0.45  # single-frame blip
    path = hmm.filter(probs / probs.sum(1, keepdims=True))
    assert (path[5:] == C).all()


def test_salami():
    txt = "# title: x\n0.0\tsilence\n1.0\tA, intro, | A:min | C:maj G:maj |\n5.0\tend\n"
    segs = salami_to_segments(txt)
    assert segs[0][2] == "N"
    assert [s[2] for s in segs[1:]] == ["A:min", "C:maj", "G:maj"]
    assert segs[-1][1] == pytest.approx(5.0)


def test_streaming_step_matches_full_forward():
    torch = pytest.importorskip("torch")
    from chordstream.config import ModelConfig
    from chordstream.model import ChordNet, StreamingStep, receptive_field_cnn

    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, context_frames=8)
    torch.manual_seed(0)
    net = ChordNet(cfg).eval()
    step = StreamingStep(net).eval()
    rf = receptive_field_cnn(cfg)
    P = cfg.n_layers * cfg.context_frames + rf  # silence warm-up flushes zero-initialised caches
    x = torch.rand(1, 40, cfg.n_bins) * 3
    seq = torch.cat([torch.zeros(1, P, cfg.n_bins), x], dim=1)
    with torch.no_grad():
        full = torch.softmax(net(seq), -1)[0, P:]
        k, v = torch.zeros(net.cache_shape()), torch.zeros(net.cache_shape())
        win = torch.zeros(1, rf, cfg.n_bins)
        outs = []
        for t in range(seq.shape[1]):
            win = torch.cat([win[:, 1:], seq[:, t : t + 1]], dim=1)
            p, k, v = step(win, k, v)
            outs.append(p[0])
    torch.testing.assert_close(torch.stack(outs[P:]), full, atol=1e-5, rtol=1e-4)


def test_banded_mask_limits_context():
    torch = pytest.importorskip("torch")
    from chordstream.config import ModelConfig
    from chordstream.model import ChordNet

    net = ChordNet(ModelConfig(d_model=32, n_layers=1, context_frames=8)).eval()
    x = torch.rand(1, 40, 144)
    x2 = x.clone()
    x2[:, :10] = 0  # outside frame 39's window (8 attn + 12 CNN frames back)
    with torch.no_grad():
        assert torch.allclose(net(x)[:, -1], net(x2)[:, -1], atol=1e-6)


def test_aligner_recovers_known_offset_and_rejects_wrong_key():
    from chordstream.data.synth import builtin_progressions, render
    from chordstream.data.youtube import align

    cfg = FrontendConfig()
    rng = np.random.default_rng(3)
    audio, segs = render(builtin_progressions()[0] * 3, rng, cfg.sample_rate, max_seconds=40)
    pad = np.zeros(int(2.4 * cfg.sample_rate), np.float32)  # upload has 2.4 s extra intro
    a = align(np.concatenate([pad, audio]), segs, cfg)
    assert abs(a["offset"] - 2.4) <= 0.08
    assert a["key_margin"] > 0.08 and a["drift"] <= 0.3
    from chordstream.vocab import CLASS_NAMES

    up = [(s, e, CLASS_NAMES[transpose(parse_harte(c), 1)] if c not in ("N", "X") else c) for s, e, c in segs]
    w = align(np.concatenate([pad, audio]), up, cfg)
    assert w["score"] < a["score"] - 0.1 and w["key_margin"] < 0.08


def test_lrc_parsing_and_title_cleanup():
    from chordstream.lyrics import clean_title, parse_lrc

    lines = parse_lrc("[00:12.50] Hello there\n[01:02.00]\n[00:03.10]First\nnot a timed line")
    assert [l["t"] for l in lines] == [3.1, 12.5, 62.0]
    assert lines[0]["text"] == "First" and lines[2]["text"] == ""  # blank = instrumental break
    assert clean_title("In the End (Official Audio)") == "In the End"
    assert clean_title("Song [Lyric Video] (Remastered 2009)") == "Song"


def test_romanize_devanagari_for_romanized_lyrics():
    pytest.importorskip("indic_transliteration")
    from chordstream.asr import romanize

    assert romanize("दिल") == "dil"  # schwa deletion: not "dila"
    assert romanize("तेरी") == "terī"
    assert romanize("hello") == "hello"


def test_transition_weights_mark_frames_after_changes():
    torch = pytest.importorskip("torch")
    from chordstream.train import transition_weights

    y = torch.tensor([[1, 1, 1, 5, 5, 5, 5, -1, -1, 5]])
    w = transition_weights(y, 3.0, 2)
    assert w.tolist() == [[1, 1, 1, 3, 3, 1, 1, 1, 1, 1]]  # change at frame 3; IGNORE gaps don't count
