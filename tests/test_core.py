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


def test_sheet_head_streaming_and_predict():
    """E26: the streaming step's sheet output equals the full forward pass of the sheet head, and
    predict_song(head="sheet") realigns by sheet_delay."""
    torch = pytest.importorskip("torch")
    from chordstream.config import ModelConfig
    from chordstream.model import ChordNet, StreamingStep, receptive_field_cnn
    from chordstream.train import _predict_song, predict_song

    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, context_frames=8, label_delay=1, sheet_delay=4)
    torch.manual_seed(0)
    net = ChordNet(cfg).eval()
    step = StreamingStep(net).eval()
    rf = receptive_field_cnn(cfg)
    P = cfg.n_layers * cfg.context_frames + rf
    seq = torch.cat([torch.zeros(1, P, cfg.n_bins), torch.rand(1, 30, cfg.n_bins) * 3], dim=1)
    with torch.no_grad():
        full = torch.softmax(net.sheet_head(net.features(seq)), -1)[0, P:]
        k, v = torch.zeros(net.cache_shape()), torch.zeros(net.cache_shape())
        win = torch.zeros(1, rf, cfg.n_bins)
        outs = []
        for t in range(seq.shape[1]):
            win = torch.cat([win[:, 1:], seq[:, t : t + 1]], dim=1)
            _, k, v, sheet = step(win, k, v)
            outs.append(sheet[0])
    torch.testing.assert_close(torch.stack(outs[P:]), full, atol=1e-5, rtol=1e-4)
    feats = seq[0].numpy()
    raw = _predict_song(net, feats, "cpu", head="sheet")
    np.testing.assert_allclose(predict_song(net, feats, "cpu", head="sheet")[:-4], raw[4:], atol=1e-6)
    assert not np.allclose(raw, _predict_song(net, feats, "cpu"))


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


def test_beat_pooling_matches_offline_definition():
    """Streaming beat chords == offline 'average posteriors over each tracker beat'."""
    from chordstream.beat import BeatTracker

    rng = np.random.default_rng(0)
    T = 300
    probs = rng.dirichlet(np.ones(25), T)
    feats = rng.random((T, 144)).astype(np.float32)
    beats = BeatTracker().run(feats)
    acc, start, streamed = np.zeros(25), 0, []
    for i, (p, b) in enumerate(zip(probs, beats)):
        acc += p
        if b.beat:
            streamed.append((start, int(acc.argmax())))
            acc[:] = 0
            start = i + 1
    bounds = [0] + [i + 1 for i, b in enumerate(beats) if b.beat]
    offline = [(a, int(probs[a:b].mean(0).argmax())) for a, b in zip(bounds[:-1], bounds[1:])]
    assert streamed == offline


def test_bidirectional_teacher_sees_future_and_chunking_is_exact():
    torch = pytest.importorskip("torch")
    from chordstream.config import ModelConfig
    from chordstream.model import ChordNet
    from chordstream.train import predict_song

    cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, context_frames=8, bidirectional=True)
    torch.manual_seed(0)
    net = ChordNet(cfg).eval()
    x = torch.rand(1, 40, 144)
    x2 = x.clone()
    x2[:, 25:] = torch.rand(1, 15, 144)
    with torch.no_grad():
        assert not torch.allclose(net(x)[:, 20], net(x2)[:, 20])  # frame 20 depends on the future
    feats = np.random.default_rng(0).random((300, 144)).astype(np.float32)
    with torch.no_grad():
        full = torch.softmax(net(torch.from_numpy(feats)[None]), -1)[0].numpy()
    np.testing.assert_allclose(predict_song(net, feats, "cpu", chunk=64), full, atol=1e-5)


def test_soft_targets_follow_pitch_shift(tmp_path):
    from chordstream.data.dataset import CropDataset
    from chordstream.vocab import CLASS_NAMES

    T = 300
    labels = np.full(T, CLASS_NAMES.index("C:maj"), np.int16)
    soft = np.zeros((T, 25), np.float16)
    soft[:, CLASS_NAMES.index("C:maj")] = 1.0
    np.savez(tmp_path / "s.npz", feats=np.random.rand(T, 144).astype(np.float16), labels=labels, soft=soft)
    ds = CropDataset([tmp_path / "s.npz"], 256, 2, augment=True, crops_per_epoch=20)
    for i in range(20):
        x, y, q, aux = ds[i]
        assert (q.argmax(1)[y >= 0] == y[y >= 0]).all()  # soft mass moves with the label


def test_extended_vocab_is_superset_of_majmin():
    from chordstream.vocab import CLASS_NAMES, N_EXT, ext_index, pretty_name, to_majmin, transpose

    assert N_EXT == 109 and CLASS_NAMES[1] == "C:maj" and CLASS_NAMES[13] == "C:min"
    assert CLASS_NAMES[ext_index(7, 2)] == "G:7" and pretty_name(ext_index(9, 4)) == "Am7"
    assert transpose(ext_index(2, 8), 5) == ext_index(7, 8)  # Dsus4 + 5 = Gsus4
    assert to_majmin(ext_index(7, 2)) == ext_index(7, 0)  # G7 -> G
    assert to_majmin(ext_index(2, 8)) == -1  # sus has no majmin triad


@pytest.mark.parametrize("label,cls,root,bass,notes", [
    ("C:maj", "C:maj", 0, 0, {0, 4, 7}),
    ("A:min7", "A:min7", 9, 9, {9, 0, 4, 7}),
    ("G:7/3", "G:7", 7, 11, {7, 11, 2, 5}),
    ("D:sus4(b7)", "D:sus4", 2, 2, {2, 7, 9, 0}),
    ("B:hdim7", "B:dim", 11, 11, {11, 2, 5, 9}),
    ("F:maj(9)", "F:maj", 5, 5, {5, 9, 0, 7}),
    ("E:9", "E:7", 4, 4, {4, 8, 11, 2, 6}),
    ("C:(1,b3,5)", "C:min", 0, 0, {0, 3, 7}),
    ("A:5", None, 9, 9, {9, 4}),  # power chord: no class, but root/notes still known
    ("C:maj/5", "C:maj", 0, 7, {0, 4, 7}),
])
def test_parse_harte_ext(label, cls, root, bass, notes):
    from chordstream.vocab import CLASS_NAMES, IGNORE, parse_harte_ext

    c, r, b, m = parse_harte_ext(label)
    assert (CLASS_NAMES[c] if c != IGNORE else None) == cls
    assert (r, b) == (root, bass)
    assert {k for k in range(12) if m >> k & 1} == notes


def test_extended_training_step_runs_and_grows_head(tmp_path):
    """One tiny training run on extended labels: structured losses, 25->109 head init."""
    torch = pytest.importorskip("torch")
    from chordstream.config import BundleConfig, FrontendConfig, ModelConfig
    from chordstream.model import ChordNet
    from chordstream.train import train
    from chordstream.vocab import N_EXT, ext_index

    small = ModelConfig(d_model=32, n_layers=1, context_frames=16)
    init = tmp_path / "init"
    init.mkdir()
    BundleConfig(FrontendConfig(), small).save(init / "config.json")
    torch.manual_seed(0)
    base = ChordNet(small)
    torch.save({"state_dict": base.state_dict()}, init / "model.pt")

    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        T = 120
        lab = np.full(T, ext_index(7, 2), np.int16)  # G7
        np.savez(cache / f"song__{i:04d}.npz", feats=rng.random((T, 144)).astype(np.float16), labels=lab,
                 root=np.full(T, 7, np.int16), bass=np.full(T, 11, np.int16),
                 notes=np.full(T, (1 << 7) | (1 << 11) | (1 << 2) | (1 << 5), np.int32))
    out = train(str(cache), str(tmp_path / "out"), epochs=1, batch_size=2, crops_per_epoch=4, init=str(init),
                workers=0, device="cpu", model_cfg=ModelConfig(n_classes=N_EXT, structured=True))
    cfg = BundleConfig.load(out / "config.json")
    assert cfg.model.n_classes == N_EXT and cfg.model.structured and cfg.model.d_model == 32
    sd = torch.load(out / "model.pt")["state_dict"] if (out / "model.pt").exists() else None
    if sd is not None:
        assert sd["head.weight"].shape[0] == N_EXT and "notes_head.weight" in sd


def test_hpss_is_causal_and_suppresses_transients():
    from chordstream.data.variants import hpss_causal

    rng = np.random.default_rng(0)
    T, F = 60, 144
    lin = np.full((T, F), 1e-3)
    lin[:, 40] = 0.5  # sustained note
    lin[30, :] += 0.5  # drum hit: broadband, one frame
    feats = np.log1p(100 * lin).astype(np.float32)
    out = hpss_causal(feats)
    assert out[30, 40] > 0.8 * feats[30, 40]  # the note survives the hit
    assert out[30, 100] < 0.5 * feats[30, 100]  # the broadband hit is suppressed
    f2 = feats.copy()
    f2[45:] = rng.random((15, F))
    np.testing.assert_allclose(hpss_causal(f2)[:45], out[:45])  # no lookahead


def test_key_input_columns_feed_the_model():
    torch = pytest.importorskip("torch")
    from chordstream.config import ModelConfig
    from chordstream.model import ChordNet

    net = ChordNet(ModelConfig(d_model=32, n_layers=1, context_frames=8, key_input=True)).eval()
    x = torch.rand(1, 20, 156)
    x2 = x.clone()
    x2[..., 144:] = torch.roll(x[..., 144:], 3, -1)
    with torch.no_grad():
        assert not torch.allclose(net(x), net(x2))


def test_delay_targets_and_stability_loss():
    torch = pytest.importorskip("torch")
    from chordstream.train import delay_targets, stability_loss

    y = torch.tensor([[1, 1, 2, 2, 2, 3]])
    q = torch.eye(25)[y.clamp_min(0)]
    aux = torch.zeros(1, 6, 3, dtype=torch.long)
    y2, q2, _ = delay_targets(y, q, aux, 2)
    assert y2.tolist() == [[-1, -1, 1, 1, 2, 2]] and torch.equal(q2[0, 2:], q[0, :4])
    steady = torch.zeros(1, 40, 25)
    steady[..., 3] = 5.0
    flicker = steady.clone()
    flicker[0, 20:22, 3], flicker[0, 20:22, 7] = 0.0, 5.0
    lab = torch.full((1, 40), 3)
    assert stability_loss(steady, lab) < 1e-6 < stability_loss(flicker, lab)


def test_label_delay_predictions_are_realigned():
    torch = pytest.importorskip("torch")
    from chordstream.config import ModelConfig
    from chordstream.model import ChordNet
    from chordstream.train import _predict_song, predict_song

    net = ChordNet(ModelConfig(d_model=32, n_layers=1, context_frames=8, label_delay=3)).eval()
    feats = np.random.default_rng(0).random((100, 144)).astype(np.float32)
    raw, aligned = _predict_song(net, feats, "cpu"), predict_song(net, feats, "cpu")
    np.testing.assert_allclose(aligned[:97], raw[3:])
