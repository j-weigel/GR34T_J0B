
# --- Audio and keyboard backends must be set BEFORE importing psychopy modules
from psychopy import prefs
prefs.hardware['audioLib']        = ['ptb', 'sounddevice', 'pygame']
prefs.hardware['keyboardBackend'] = 'iohub'   # required for egi_pynetstation;
                                               # prevents b'ZZ' errors from PTB
                                               # keyboard interfering with ECI

# ── Audio device selection ──────────────────────────────────────────────────
# On the lab run computer: route to Realtek headphones.
# On any other machine (students' laptops): fall back to whatever
# PsychoPy finds first, which is the correct default behaviour.
_PREFERRED_AUDIO_SUBSTR = 'Realtek'

def _find_output_device(substr):
    """Return the first output device name containing substr, or None."""
    try:
        import sounddevice as sd
        for d in sd.query_devices():
            if (d.get('max_output_channels', 0) > 0 and
                    substr.lower() in d.get('name', '').lower()):
                return d['name']
    except Exception:
        pass
    return None

_audio_device = _find_output_device(_PREFERRED_AUDIO_SUBSTR)
if _audio_device is not None:
    prefs.hardware['audioDevice'] = _audio_device
# If None: leave audioDevice unset → PsychoPy selects the first available
# output device automatically, which is what students' laptops need.
# ────────────────────────────────────────────────────────────────────────────
prefs.hardware['audioLatencyMode'] = '3'  # aggressive low-latency
# ---------------------------------------------------------------------------

import os
import sys
import csv
import json
import hashlib
import logging as py_logging
from datetime import datetime

import numpy as np
import pandas as pd

from psychopy import visual, core, event, gui, sound, logging, data


# ===========================================================================
# CONFIG  ——  All knobs students are likely to want to change live here.
# ===========================================================================


# Stimulus presentation timings (seconds)
STUDY_MARKER_RADIUS       = 5     # Red dot at image centre during study AND test response marker
INITIAL_SETTLING_SEC      = 5.0   # Delay after operator starts cueing phase
WORD_POS_Y                = 360   # Label y-position: safely above the highest possible item
                                   # (max item y = half_h - SCREEN_PADDING = 300 for default settings)
SCREEN_PADDING            = 100   # Keep randomized items away from screen edges

N_STUDY_REPEATS           = 2     # how many times each item is shown in study
STUDY_ITEM_DURATION       = 2.5   # total time the image is visible each study trial
STUDY_SOUND_ONSET         = 0.25  # delay from image onset until sound plays
STUDY_ITI                 = 0.75  # blank inter-trial interval

# Memory test (used for BOTH pre- and post-test)
MARKER_START_POS          = (0, 0)   # marker starts at screen center each trial
TEST_MAX_TRIAL_SEC        = 20.0     # give up after this many seconds per item

# Cue eligibility: items with pre-test error ≤ this are "well enough learned"
# and eligible for the cued/uncued random split. In pixels. Tune for screen size.

# Default is deliberately liberal (half the screen diagonal ≈ 385 px for
# 1280×800) so that most items qualify in a short demo or pilot. Tighten
# this once you have pilot data showing the actual error distribution.
# See README for a full tuning guide.
CUE_ELIGIBILITY_ERROR_MAX_PX = 350

# Fraction of well-learned items randomly assigned to the CUED group
CUED_FRACTION             = 0.5

# Cueing phase
DEFAULT_MAX_CUE_REPS      = 3     # max times each eligible item is replayed
INTER_CUE_INTERVAL_SEC    = (4.5, 5.5)   # uniform jitter between cue onsets
DEFAULT_SETTLING_SEC      = 15.0  # post-arousal wait before cues resume
CUEING_VOLUME             = 0.5   # 0.0–1.0; keep low for sleeping participant

# Display
SCREEN_SIZE               = (1280, 800)
SCREEN_UNITS              = 'pix'
BG_COLOR                  = 'black'
FG_COLOR                  = 'white'

# Operator keys (lowercase; PsychoPy reports these as-is unless modifiers used)
KEY_AROUSAL               = 'a'
KEY_PAUSE                 = 'p'
KEY_QUIT                  = 'q'
KEY_JUMP_POSTTEST         = 'j'
KEY_RESUME                = 'r'
KEY_SKIP_ITEM             = 'k'
KEY_CUE_BEGIN             = 'space'
KEY_CONFIRM               = 'space'
KEY_ESC                   = 'escape'

# EEG / NetStation  ---------------------------------------------------------
# Two APIs are supported. Flip EEG_API to match your lab's setup.
#   'old'  psychopy.hardware.egi (ships with PsychoPy; legacy; works on
#          most existing EGI lab setups; minimal active maintenance).
#   'new'  egi_pynetstation (pip install egi-pynetstation; required for
#          newer NetStation / Net Amps 400+ that deprecated the old protocol;
#          requires an NTP-reachable host — usually the same machine running
#          NetStation).
# Either way NetStation enforces 4-character event keys. See the TRIG dict.
EEG_MODE                  = False
EEG_API                   = 'new'            # 'old' or 'new'
EEG_HOST                  = '10.10.10.42'   # IP of the NetStation acquisition PC
EEG_PORT                  = 55513           # NetStation ECI port
EEG_NTP_HOST              = '10.10.10.51'   # Only used when EEG_API == 'new';
                                             # This is the IP_amp (amplifier's IP)

# 4-char trigger codes. Keep them mnemonic; NetStation truncates anything
# longer. `label` fields in send_trigger() carry the human-readable name.
TRIG = {
    'study_item_onset'   : 'STDI',
    'study_sound_onset'  : 'STDS',
    'pretest_onset'      : 'PRE+',
    'pretest_response'   : 'PRER',
    'cue_onset'          : 'CUE+',
    'arousal_flag'       : 'ARSL',
    'pause_flag'         : 'PAUS',
    'resume_from_int'    : 'RESU',
    'jump_to_posttest'   : 'JUMP',
    'posttest_onset'     : 'POS+',
    'posttest_response'  : 'POSR',
    'phase_boundary'     : 'PHAS',
}

# Grid (visible spatial reference frame)
# Drawn in study AND test by default so encoding and recall happen against
# the SAME visible reference. For closer fidelity to the original Rudoy
# (2009) task (blank background during encoding), set SHOW_GRID_IN_STUDY
# to False while keeping SHOW_GRID_IN_TEST = True. Grid opacity is low so
# any `background_stims` you pass into run_memory_test() remain visible.
SHOW_GRID_IN_STUDY        = True
SHOW_GRID_IN_TEST         = True
GRID_SPACING_PX           = 225
GRID_COLOR                = 'gray'
GRID_OPACITY              = 0.7
GRID_LINE_WIDTH           = 2

# Mouse response marker (memory test) — same radius as the study anchor dot
# so the participant is matching "dot to remembered dot" consistently.

# Files
DATA_DIR                  = 'data'
DEFAULT_TRIAL_FILE        = 'stimuli/trials_object_locationNEW.csv'


# ===========================================================================
# MODULE-LEVEL STATE  —  populated in main(); kept at module scope so that
# helper functions (send_trigger, logging writers) can reach it without
# threading state through every call. Students: avoid reassigning these.
# ===========================================================================

_ns = None                    # NetStation handle (or None)
_ns_is_new_api = False        # Tracks which EEG_API was selected at connect time
_session_info = {}            # populated by run_setup_dialog
_event_log = []               # every send_trigger() appends here
_run_dir = None               # per-run output directory


# ===========================================================================
# HELPERS
# ===========================================================================

def make_seed(subject_number: int, session: str) -> int:
    """Deterministic 32-bit seed from (subject, session). Stable across OSes."""
    s = f"{subject_number}__{session}".encode('utf-8')
    return int(hashlib.md5(s).hexdigest()[:8], 16)


def ensure_run_dir(subject_number: int, session: str) -> str:
    """Create (if needed) and return the per-run output directory."""
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f"sub-{subject_number:03d}_ses-{session}_{stamp}"
    path = os.path.join(DATA_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def write_csv(rows, path):
    """Write a list-of-dicts to CSV, union of all keys. No-op on empty input."""
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def send_trigger(trig_key: str, label: str = None, meta: dict = None):
    """
    Log an event to the run event log and (if EEG_MODE) forward to NetStation.

    trig_key : one of the values in TRIG (4-char NetStation key)
    label    : human-readable description
    meta     : dict of small metadata fields (item_id, rep, etc.)

    The event is ALWAYS appended to the in-memory _event_log (which is written
    to event_log.csv at end of run, even on crash). If EEG_MODE is on and the
    connection succeeded, the event is additionally forwarded to NetStation
    via the API selected by EEG_API. A failed send warns but does NOT abort
    the experiment — a dropped trigger is less costly than an aborted run.
    """
    t = core.getTime()
    if meta is None:
        meta = {}
    rec = {
        'time_sec': t,
        'trig_key': trig_key,
        'label'   : label or trig_key,
    }
    for k, v in meta.items():
        rec[f'meta_{k}'] = v
    _event_log.append(rec)

    if EEG_MODE and _ns is not None:
        try:
            # NetStation event keys are 4 chars — truncate/pad defensively.
            key4 = (trig_key + '____')[:4]
            desc_str = (label or trig_key)[:80]
            if _ns_is_new_api:
                # egi_pynetstation: use desc= (not description=) and omit
                # table= — matches the proven working BNB_MMN pattern.
                _ns.send_event(
                    event_type=key4,
                    label=desc_str,
                    desc=desc_str,
                    start=0.0,
                )
            else:
                # psychopy.hardware.egi signature: key, label, timestamp, table.
                tbl = {str(k)[:4]: str(v)[:8] for k, v in meta.items()}
                _ns.send_event(
                    key=key4,
                    label=desc_str,
                    timestamp=None,
                    table=tbl,
                )
        except Exception as e:
            # Missed triggers are annoying but must NOT kill the experiment.
            logging.warning(f'send_trigger failed for {trig_key}: {e}')


def resync_eeg():
    """Re-synchronize the NTP clock with the amplifier.

    Call at the start of each phase AND before each individual trial/cue.
    egi_pynetstation requires periodic resync to keep timestamps aligned
    with the amplifier's clock — without this, the TCP connection goes idle
    between events, send_event() fails silently, and triggers disappear.
    Safe no-op if EEG_MODE is off or _ns is None.
    """
    if not EEG_MODE or _ns is None:
        return
    try:
        if _ns_is_new_api:
            _ns.resync()
        else:
            _ns.sync()
    except Exception as e:
        logging.warning(f'resync_eeg failed: {e}')


def send_trigger_on_flip(win, trig_key: str, label: str = None,
                         meta: dict = None):
    """Schedule a trigger to fire at the next win.flip() — screen-synced.

    Use for events time-locked to a visual onset (study image onset, test
    probe onset). For non-visual events (audio cue onset, phase boundaries)
    use send_trigger() directly.
    """
    def _callback():
        send_trigger(trig_key, label, meta)
    win.callOnFlip(_callback)


def connect_netstation_if_enabled():
    """
    Connect to NetStation if EEG_MODE. Safe no-op otherwise.

    Honors EEG_API:
      'old' — psychopy.hardware.egi: Netstation(host,port) → connect() →
              BeginSession() → sync() → StartRecording()
      'new' — egi_pynetstation:      NetStation(host,port) → connect(ntp_ip) →
              begin_rec()

    A connection failure logs an error and leaves _ns = None, so the rest
    of the experiment runs normally — triggers still land in event_log.csv.
    """
    global _ns, _ns_is_new_api
    if not EEG_MODE:
        return

    if EEG_API == 'new':
        try:
            # Requires: pip install egi-pynetstation
            from egi_pynetstation.NetStation import NetStation
            _ns = NetStation(EEG_HOST, EEG_PORT)
            _ns.connect(ntp_ip=EEG_NTP_HOST)
            _ns.begin_rec()
            _ns_is_new_api = True
            # Confirm connection works with an initial event (matches BNB pattern).
            _ns.send_event(event_type='STRT', label='ExperimentStart',
                           desc='ExperimentStart', start=0.0)
            logging.info(f'NetStation (new API) connected and recording '
                         f'({EEG_HOST}:{EEG_PORT}, NTP {EEG_NTP_HOST})')
        except Exception as e:
            logging.error(f'NetStation (new API) connection FAILED: {e}. '
                          'Continuing without EEG triggers.')
            _ns = None
    else:  # 'old' or anything else → safe fallback to old API
        try:
            from psychopy.hardware import egi
            _ns = egi.Netstation(host=EEG_HOST, port=EEG_PORT)
            _ns.connect()
            _ns.BeginSession()
            _ns.sync()
            _ns.StartRecording()
            _ns_is_new_api = False
            logging.info(f'NetStation (old API) connected and recording '
                         f'({EEG_HOST}:{EEG_PORT})')
        except Exception as e:
            logging.error(f'NetStation (old API) connection FAILED: {e}. '
                          'Continuing without EEG triggers.')
            _ns = None


def disconnect_netstation():
    """Cleanly end recording and disconnect. Honors the API used at connect."""
    global _ns
    if _ns is None:
        return
    try:
        if _ns_is_new_api:
            _ns.end_rec()
            _ns.disconnect()
        else:
            _ns.StopRecording()
            _ns.EndSession()
            _ns.disconnect()
    except Exception as e:
        logging.warning(f'NetStation shutdown issue: {e}')
    _ns = None


def show_text_wait(win, txt, wait_keys=('space',), height=28, timeout=None):
    """Draw centered text until one of wait_keys is pressed. Returns the key."""
    ts = visual.TextStim(win, text=txt, color=FG_COLOR, height=height,
                         wrapWidth=SCREEN_SIZE[0] * 0.85, alignText='center')
    ts.draw()
    win.flip()
    event.clearEvents()
    clock = core.Clock()
    keys_to_wait = list(wait_keys) + [KEY_ESC]
    while True:
        keys = event.getKeys(keyList=keys_to_wait)
        if KEY_ESC in keys:
            raise KeyboardInterrupt("Operator pressed ESC")
        if keys:
            return keys[0]
        if timeout is not None and clock.getTime() > timeout:
            return None
        core.wait(0.01)


def flash_text(win, txt, dur=1.5, height=28):
    """Non-blocking style message for brief operator feedback."""
    ts = visual.TextStim(win, text=txt, color=FG_COLOR, height=height,
                         wrapWidth=SCREEN_SIZE[0] * 0.85, alignText='center')
    ts.draw()
    win.flip()
    core.wait(dur)


def build_grid_stims(win):
    """
    Construct a list of Line stims forming a reference grid centered at (0,0).

    Returned stims span the window at GRID_SPACING_PX intervals. Call this
    ONCE during setup — constructing Line stims is cheap at rest but doing
    it every frame wastes GPU/CPU.

    Drawing order convention:
        background_stims → grid → foreground (markers, text)

    Returns an empty list when both SHOW_GRID_IN_STUDY and SHOW_GRID_IN_TEST
    are False.
    """
    if not (SHOW_GRID_IN_STUDY or SHOW_GRID_IN_TEST):
        return []
    stims = []
    half_w, half_h = SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2

    # Vertical lines, centered on x=0 so the grid is symmetric
    xs = {0.0}
    x = float(GRID_SPACING_PX)
    while x <= half_w:
        xs.add(x); xs.add(-x)
        x += GRID_SPACING_PX
    for x in sorted(xs):
        stims.append(visual.Line(
            win, start=(x, -half_h), end=(x, half_h),
            lineColor=GRID_COLOR, lineWidth=GRID_LINE_WIDTH,
            opacity=GRID_OPACITY, units=SCREEN_UNITS))

    # Horizontal lines, centered on y=0
    ys = {0.0}
    y = float(GRID_SPACING_PX)
    while y <= half_h:
        ys.add(y); ys.add(-y)
        y += GRID_SPACING_PX
    for y in sorted(ys):
        stims.append(visual.Line(
            win, start=(-half_w, y), end=(half_w, y),
            lineColor=GRID_COLOR, lineWidth=GRID_LINE_WIDTH,
            opacity=GRID_OPACITY, units=SCREEN_UNITS))

    return stims


def draw_grid(grid_stims):
    """Draw prebuilt grid lines. No-op for None/empty."""
    if not grid_stims:
        return
    for s in grid_stims:
        s.draw()


# ===========================================================================
# SETUP DIALOG & EXPERIMENTER INSTRUCTIONS
# ===========================================================================

def run_setup_dialog():
    """Gather session parameters from the operator. Returns a dict."""
    info = {
        'subject_number'     : 1,
        'session'            : 'pilot',
        'trial_file'         : DEFAULT_TRIAL_FILE,
        'max_cue_repetitions': DEFAULT_MAX_CUE_REPS,
        'settling_interval'  : DEFAULT_SETTLING_SEC,
        'cueing_target'      : ['sleep', 'wake_silent', 'wake_passive'],
        # --- Cue eligibility ---
        # 'learned_by_pretest'  cue the half of items the participant got right
        #                       (error <= threshold below). Classic Rudoy design.
        # 'unlearned_by_pretest' cue items the participant got WRONG at pre-test.
        #                       Classic Schreiner/Rasch design.
        # 'all_study_items'     ignore pre-test performance; cue a random half
        #                       of everything that was studied.
        'cue_mode'           : ['learned_by_pretest',
                                'unlearned_by_pretest',
                                'all_study_items'],
        'cue_threshold_px'   : CUE_ELIGIBILITY_ERROR_MAX_PX,
        'start_phase'        : ['study', 'pre_test', 'cueing', 'post_test'],
        'fullscreen'         : False,
        'eeg_mode'           : EEG_MODE,
    }
    order = ['subject_number', 'session', 'trial_file',
             'max_cue_repetitions', 'settling_interval',
             'cueing_target', 'cue_mode', 'cue_threshold_px',
             'start_phase', 'fullscreen', 'eeg_mode']
    dlg = gui.DlgFromDict(info, title='TMR: Object–Location–Sound', order=order)
    if not dlg.OK:
        core.quit()
    # Validate & coerce
    try:
        info['subject_number'] = int(info['subject_number'])
    except ValueError:
        raise SystemExit('subject_number must be an integer.')
    info['max_cue_repetitions'] = int(info['max_cue_repetitions'])
    info['settling_interval']   = float(info['settling_interval'])
    info['cue_threshold_px']    = float(info['cue_threshold_px'])
    info['seed'] = make_seed(info['subject_number'], info['session'])
    return info


EXPERIMENTER_BRIEFING = """\
OPERATOR BRIEFING — Object–Location–Sound TMR

Start-phase:
  study     full session from the beginning
  pre_test  skip study (participant already studied)
  cueing    skip study + pre-test (use for resume-from-crash or pilot)
  post_test jump straight to post-test

Settling interval:
  After ANY classified interruption during cueing, cues pause for this many
  seconds before resuming. Default 15 s. Set 0 to resume immediately.

Keys during cueing:
  A   flag arousal (will prompt to classify and choose an action)
  P   operator-initiated pause (same prompts)
  J   jump to post-test immediately
  Q   quit experiment

After A/P you will be prompted to:
  1. CLASSIFY the event (0–5)
  2. Choose an ACTION (R/K/J/Q)

Cueing volume is low by default (see CUEING_VOLUME in config) — appropriate
for a sleeping participant. Re-check at each session; speaker placement
and ambient noise change what "low" means.

Press SPACE to continue.
"""


# ===========================================================================
# STIMULUS LOADING
# ===========================================================================

REQUIRED_COLS = {'item_id', 'word', 'image_file', 'sound_file',
                 'target_x', 'target_y'}



def get_or_create_trials(info):
    """Checks for existing randomized trials for this sub; creates them if not found.

    Saves a per-subject CSV with randomized target_x/target_y to DATA_DIR so
    that the same participant always sees the same locations — even if the
    experiment is restarted between study and test (e.g., after a crash).

    Paths in the saved CSV are stored as ABSOLUTE paths so that load_trials()
    resolves them correctly regardless of which directory the CSV lives in.
    """
    sub_trial_file = os.path.join(DATA_DIR, f"sub-{info['subject_number']:03d}_trials.csv")

    # Resolve the original trial file's directory once, for path expansion below.
    orig_base = os.path.dirname(os.path.abspath(info['trial_file']))

    if os.path.exists(sub_trial_file):
        logging.info(f"Loading existing randomized trials from {sub_trial_file}")
        # Paths in the saved file are already absolute (written that way below),
        # so load_trials will resolve them correctly.
        return load_trials(sub_trial_file)

    logging.info(f"Generating new randomized locations for subject {info['subject_number']}")
    base_df = pd.read_csv(info['trial_file'])
    rng = np.random.default_rng(info['seed'])

    half_w = (SCREEN_SIZE[0] / 2) - SCREEN_PADDING
    half_h = (SCREEN_SIZE[1] / 2) - SCREEN_PADDING

    base_df['target_x'] = rng.uniform(-half_w, half_w, size=len(base_df)).round(1)
    base_df['target_y'] = rng.uniform(-half_h, half_h, size=len(base_df)).round(1)

    # Convert image_file and sound_file to absolute paths before saving.
    # load_trials() resolves relative paths relative to the CSV's own directory,
    # so if we save the per-subject CSV to data/ with relative paths like
    # "images/cat.png", it would look for data/images/cat.png — wrong.
    # Storing absolute paths sidesteps this entirely.
    for col in ('image_file', 'sound_file'):
        base_df[col] = base_df[col].apply(
            lambda p: p if os.path.isabs(str(p))
                      else os.path.abspath(os.path.join(orig_base, str(p))))

    os.makedirs(DATA_DIR, exist_ok=True)
    base_df.to_csv(sub_trial_file, index=False)
    return load_trials(sub_trial_file)

def load_trials(trial_file: str):

    """
    Read the trial CSV. Each row = one unique item. Columns:
      item_id, word, image_file, sound_file, target_x, target_y
    Paths are interpreted relative to the trial file's directory.
    """
    if not os.path.isfile(trial_file):
        raise FileNotFoundError(f'Trial file not found: {trial_file}')
    df = pd.read_csv(trial_file)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f'Trial file missing required columns: {missing}')

    base = os.path.dirname(os.path.abspath(trial_file))
    trials = []
    for _, row in df.iterrows():
        t = dict(row)
        # Resolve relative paths
        for col in ('image_file', 'sound_file'):
            if not os.path.isabs(t[col]):
                t[col] = os.path.join(base, t[col])
        t['target_x'] = float(t['target_x'])
        t['target_y'] = float(t['target_y'])
        t['item_id']  = str(t['item_id'])
        trials.append(t)

    # Validate existence — warn but don't die (so code can be tested w/o assets)
    missing_files = []
    for t in trials:
        for col in ('image_file', 'sound_file'):
            if not os.path.isfile(t[col]):
                missing_files.append((t['item_id'], col, t[col]))
    if missing_files:
        logging.warning(f'{len(missing_files)} stimulus files missing. '
                        f'First few: {missing_files[:5]}')
    return trials


# ===========================================================================
# STUDY PHASE
# ===========================================================================

def run_study_phase(win, trials, info, grid_stims=None):
    """Show each item at its target location with paired sound.

    grid_stims : list from build_grid_stims() to overlay as a reference
                 frame. Drawn BEFORE the item image, so the image stays on
                 top of the grid. Controlled by SHOW_GRID_IN_STUDY.
    """
    resync_eeg()   # re-align NTP clock before this phase
    send_trigger(TRIG['phase_boundary'], 'study_start')

    show_text_wait(win, (
        "STUDY PHASE\n\n"
        "Images will appear with a red dot at their center.\n"
        "Please learn the location of the RED DOT for each object.\n"
        "The name of the object will appear at the top of the screen.\n\n"
        "Press SPACE to begin."))

    rng = np.random.default_rng(info['seed'])
    trial_sequence = []
    for rep in range(N_STUDY_REPEATS):
        order = list(range(len(trials)))
        rng.shuffle(order)
        for idx in order:
            trial_sequence.append((rep, trials[idx]))

    # Pre-load reusable stims
    img = visual.ImageStim(win, image=None, units=SCREEN_UNITS, size=(120, 120))
    label = visual.TextStim(win, text='', color=FG_COLOR, units=SCREEN_UNITS,
                            pos=(0, WORD_POS_Y), height=30)
    marker_anchor = visual.Circle(win, radius=STUDY_MARKER_RADIUS, 
                                  fillColor='red', lineColor='white', units=SCREEN_UNITS)

    study_log = []
    for study_idx, (rep, trial) in enumerate(trial_sequence):
        resync_eeg()  # keep NetStation socket alive; resync before every trial
        try:
            img.image = trial['image_file']
            img.pos = (trial['target_x'], trial['target_y'])
        except Exception as e:
            logging.warning(f"Image load failed for {trial['item_id']}: {e}")

        label.text = trial['word']
        marker_anchor.pos = (trial['target_x'], trial['target_y'])

        try:
            snd = sound.Sound(trial['sound_file'])
            snd.setVolume(0.7)
        except Exception as e:
            logging.warning(f"Sound load failed for {trial['item_id']}: {e}")
            snd = None

        if SHOW_GRID_IN_STUDY:
            draw_grid(grid_stims)
        img.draw()
        label.draw()
        marker_anchor.draw()
        send_trigger_on_flip(win, TRIG['study_item_onset'],
                     f"study_item_{trial['item_id']}",
                     {'item_id': trial['item_id'], 'rep': rep})
        t_onset = win.flip()

        core.wait(STUDY_SOUND_ONSET)
        if snd is not None:
            snd.play()
            send_trigger(TRIG['study_sound_onset'],
                         f"study_sound_{trial['item_id']}",
                         {'item_id': trial['item_id'], 'rep': rep})

        core.wait(STUDY_ITEM_DURATION - STUDY_SOUND_ONSET)
        if snd is not None:
            try:
                snd.stop()
            except Exception:
                pass

        if SHOW_GRID_IN_STUDY:
            draw_grid(grid_stims)
        win.flip()
        core.wait(STUDY_ITI)

        study_log.append({
            'study_index': study_idx,
            'rep': rep,
            'item_id': trial['item_id'],
            'word': trial['word'],
            'target_x': trial['target_x'],
            'target_y': trial['target_y'],
            'onset_time_sec': t_onset,
        })

        if KEY_ESC in event.getKeys():
            raise KeyboardInterrupt("Operator pressed ESC during study")

    write_csv(study_log, os.path.join(_run_dir, 'study_trials.csv'))
    send_trigger(TRIG['phase_boundary'], 'study_end')
    return study_log


def run_memory_test(win, trials, info, phase_label: str,
                    grid_stims=None, background_stims=None):
    """
    Pre- or post-test: participant drags a marker to the remembered location.

    Parameters
    ----------
    phase_label      : 'pre_test' or 'post_test'
    grid_stims       : list from build_grid_stims(); drawn BEHIND the response
                       marker but IN FRONT of background_stims. Controlled
                       by SHOW_GRID_IN_TEST.
    background_stims : optional list of visual stims drawn FIRST (behind the
                       grid and marker). Use this to inject task-specific
                       visual context without modifying this function —
                       e.g., persistent anchor images, colored regions,
                       reference landmarks, condition-specific cues.

    Draw order each frame (back → front):
        background_stims  →  grid_stims  →  word (top label)  →  marker

    Grid opacity is low (GRID_OPACITY) so background_stims remain visible
    through it. If you need opaque background content, set GRID_OPACITY
    lower still (or set SHOW_GRID_IN_TEST = False).

    Response protocol
    -----------------
        Hold LEFT MOUSE BUTTON  →  marker follows cursor (drag)
        Release LMB             →  marker stays at last position
        Press SPACE             →  confirm response and advance
        Press ESC               →  abort the session

    Returns
    -------
    List of per-trial dicts including target_x/y, resp_x/y, error_px, rt_sec.
    """
    resync_eeg()   # re-align NTP clock before this phase
    send_trigger(TRIG['phase_boundary'], f'{phase_label}_start')

    show_text_wait(win,
        f"{phase_label.upper().replace('_',' ')}\n\n"
        "A word will appear at the top of the screen.\n\n"
        "Use the MOUSE to drag the red marker to the location\n"
        "you remember that object being during study.\n"
        "Hold the LEFT BUTTON to move the marker; release to let go.\n\n"
        "Press SPACE to confirm each response.\n\n"
        "Press SPACE to begin.")

    marker = visual.Circle(win, radius=STUDY_MARKER_RADIUS,
                           fillColor='red', lineColor='white',
                           units=SCREEN_UNITS)
    word = visual.TextStim(win, text='', color=FG_COLOR, units=SCREEN_UNITS,
                           height=36, pos=(0, SCREEN_SIZE[1] / 2 - 50))

    mouse = event.Mouse(win=win, visible=True)

    # Jitter order with a phase-specific seed so pre ≠ post but both are stable
    rng = np.random.default_rng(info['seed'] + (1 if phase_label == 'pre_test' else 2))
    order = list(range(len(trials)))
    rng.shuffle(order)

    results = []
    trig_onset = TRIG['pretest_onset']    if phase_label == 'pre_test' else TRIG['posttest_onset']
    trig_resp  = TRIG['pretest_response'] if phase_label == 'pre_test' else TRIG['posttest_response']

    show_grid = SHOW_GRID_IN_TEST and bool(grid_stims)

    for i, idx in enumerate(order):
        trial = trials[idx]
        resync_eeg()  # keep NetStation socket alive; resync before every trial
        word.text = trial['word']
        marker.pos = MARKER_START_POS
        try:
            mouse.setPos(MARKER_START_POS)
        except Exception:
            # Some OS / backend combinations disallow programmatic cursor
            # repositioning; it's a nicety, not a requirement.
            pass
        mouse.clickReset()

        t_onset = core.getTime()
        send_trigger(trig_onset, f"{phase_label}_item_{trial['item_id']}",
                     {'item_id': trial['item_id'], 'trial_idx': i})

        confirmed = False
        timed_out = False
        event.clearEvents()

        while not confirmed:
            # --- Draw back-to-front ---
            if background_stims:
                for s in background_stims:
                    s.draw()
            if show_grid:
                draw_grid(grid_stims)
            word.draw()
            marker.draw()
            win.flip()

            # --- Mouse: marker follows cursor while LMB is held ---
            if mouse.getPressed()[0]:
                marker.pos = mouse.getPos()

            # --- Keyboard ---
            keys = event.getKeys(keyList=[KEY_CONFIRM, KEY_ESC])
            if KEY_ESC in keys:
                raise KeyboardInterrupt(f"Operator ESC during {phase_label}")
            if KEY_CONFIRM in keys:
                confirmed = True

            # --- Per-trial timeout ---
            if core.getTime() - t_onset > TEST_MAX_TRIAL_SEC:
                timed_out = True
                break

        rt = core.getTime() - t_onset
        resp_x, resp_y = float(marker.pos[0]), float(marker.pos[1])
        err = float(np.hypot(resp_x - trial['target_x'], resp_y - trial['target_y']))

        send_trigger(trig_resp, f"{phase_label}_resp_{trial['item_id']}",
                     {'item_id': trial['item_id'], 'err': round(err, 1)})

        results.append({
            'phase'      : phase_label,
            'trial_idx'  : i,
            'item_id'    : trial['item_id'],
            'word'       : trial['word'],
            'target_x'   : trial['target_x'],
            'target_y'   : trial['target_y'],
            'resp_x'     : resp_x,
            'resp_y'     : resp_y,
            'error_px'   : err,
            'rt_sec'     : rt,
            'timed_out'  : timed_out,
        })

    write_csv(results, os.path.join(_run_dir, f'{phase_label}_responses.csv'))
    send_trigger(TRIG['phase_boundary'], f'{phase_label}_end')
    return results


# ===========================================================================
# CUE ASSIGNMENT  —  within-subject random split
# ===========================================================================

def assign_cue_groups(trials, pretest_results, info):
    """
    Classify each trial into a cue group based on info['cue_mode']:

      'learned_by_pretest'   (default / Rudoy-style)
          Eligible pool = items whose pre-test error ≤ info['cue_threshold_px'].
          Half of the pool → 'cued_well_learned', other half → 'uncued_well_learned'.
          Items outside the threshold → 'not_well_learned'.

      'unlearned_by_pretest'   (Schreiner/Rasch-style)
          Eligible pool = items whose pre-test error > info['cue_threshold_px']
          (i.e., those the participant got wrong / placed far from target).
          Half → 'cued_unlearned', half → 'uncued_unlearned'.
          Items within the threshold (already learned) → 'already_learned'.

      'all_study_items'
          Ignore pre-test performance entirely. All study items enter the pool.
          Half → 'cued_all', half → 'uncued_all'.
          No items are excluded.

    If pretest_results is empty (no pre-test was run in this session, e.g.
    start_phase='cueing'), ALL modes fall back to splitting all items randomly
    with 'fallback_cued' / 'fallback_uncued' labels.

    Returns: dict  item_id -> group_label
    """
    rng    = np.random.default_rng(info['seed'] + 500)
    mode   = info.get('cue_mode', 'learned_by_pretest')
    thresh = float(info.get('cue_threshold_px', CUE_ELIGIBILITY_ERROR_MAX_PX))
    groups = {}

    all_ids = [t['item_id'] for t in trials]

    # ── Fallback if no pre-test data ─────────────────────────────────────
    if not pretest_results or mode == 'all_study_items':
        ids = list(all_ids)
        rng.shuffle(ids)
        n_cue = int(round(CUED_FRACTION * len(ids)))
        if mode == 'all_study_items':
            for i, iid in enumerate(ids):
                groups[iid] = 'cued_all' if i < n_cue else 'uncued_all'
        else:
            for i, iid in enumerate(ids):
                groups[iid] = 'fallback_cued' if i < n_cue else 'fallback_uncued'
        return groups

    # ── Pre-test-based modes ──────────────────────────────────────────────
    pre_by_id = {r['item_id']: r for r in pretest_results}
    eligible   = []

    for iid in all_ids:
        pre = pre_by_id.get(iid)
        err = pre['error_px'] if pre is not None else float('inf')

        if mode == 'learned_by_pretest':
            if err <= thresh:
                eligible.append(iid)
            else:
                groups[iid] = 'not_well_learned'

        elif mode == 'unlearned_by_pretest':
            if err > thresh:
                eligible.append(iid)
            else:
                groups[iid] = 'already_learned'

    rng.shuffle(eligible)
    n_cue = int(round(CUED_FRACTION * len(eligible)))

    if mode == 'learned_by_pretest':
        for i, iid in enumerate(eligible):
            groups[iid] = 'cued_well_learned' if i < n_cue else 'uncued_well_learned'
    elif mode == 'unlearned_by_pretest':
        for i, iid in enumerate(eligible):
            groups[iid] = 'cued_unlearned' if i < n_cue else 'uncued_unlearned'

    return groups


def run_cueing_phase(win, trials, info, cue_groups):
    """
    Play sound cues for items in the cued group, interleaved over up to
    max_cue_repetitions rounds. Screen order:
      1. Item-count summary (operator sees how many items qualified FIRST)
      2. Sleep stability check (operator confirms N3)
      3. Settling countdown
      4. Cueing loop

    Returns (cue_log, interruption_log, end_state): 'completed','jump_post','quit'.
    """
    resync_eeg()   # re-align NTP clock before this phase
    send_trigger(TRIG['phase_boundary'], 'cueing_start')

    # All group labels that mean "this item should be cued"
    CUED_LABELS   = {'cued_well_learned', 'cued_unlearned',
                     'cued_all', 'fallback_cued'}
    UNCUED_LABELS = {'uncued_well_learned', 'uncued_unlearned',
                     'uncued_all', 'fallback_uncued'}

    cued_ids       = [iid for iid, g in cue_groups.items() if g in CUED_LABELS]
    uncued_ids     = [iid for iid, g in cue_groups.items() if g in UNCUED_LABELS]
    ineligible_ids = [iid for iid, g in cue_groups.items()
                      if g not in CUED_LABELS | UNCUED_LABELS]
    fallback       = any(g.startswith('fallback_') for g in cue_groups.values())
    total          = len(cue_groups)
    mode           = info.get('cue_mode', 'learned_by_pretest')
    thresh         = float(info.get('cue_threshold_px', CUE_ELIGIBILITY_ERROR_MAX_PX))

    # ── STEP 1: Item-count summary (BEFORE the timer starts) ─────────────
    warn_line = (
        "\n⚠  0 items to cue! Check cue_mode and cue_threshold_px.\n"
        f"   Raise threshold (currently {thresh:.0f} px) or switch to\n"
        "   'all_study_items' mode.\n"
        if len(cued_ids) == 0 else "")
    show_text_wait(win,
        f"CUEING PHASE — item eligibility summary\n\n"
        f"Cue mode:                 {mode}\n"
        f"Eligibility threshold:    {thresh:.0f} px\n\n"
        f"Items total:              {total}\n"
        f"  → CUED:                 {len(cued_ids)}\n"
        f"  → UNCUED (control):     {len(uncued_ids)}\n"
        f"  → Excluded:             {len(ineligible_ids)}\n"
        f"  → Fallback mode:        {fallback}\n"
        f"{warn_line}\n"
        f"During cueing: A=arousal  P=pause  J=jump-to-post  Q=quit\n\n"
        f"Press SPACE to continue to the sleep-stability check.",
        wait_keys=[KEY_CUE_BEGIN])

    # ── STEP 2: Sleep stability check ────────────────────────────────────
    show_text_wait(win, (
        "SLEEP STABILITY CHECK\n\n"
        "1. Verify participant is in N3 (Slow Wave Sleep).\n"
        "2. Ensure at least 2 minutes of stable N3 have passed.\n"
        "3. Check for absence of arousals or spindles.\n\n"
        "Press SPACE to begin the settling period and start cueing."))

    # ── STEP 3: Settling countdown (visible feedback) ─────────────────────
    if INITIAL_SETTLING_SEC > 0:
        settling_msg = visual.TextStim(
            win, text='', color=FG_COLOR, height=28,
            wrapWidth=SCREEN_SIZE[0] * 0.8, alignText='center')
        t_settle_start = core.getTime()
        while True:
            elapsed   = core.getTime() - t_settle_start
            remaining = INITIAL_SETTLING_SEC - elapsed
            if remaining <= 0:
                break
            settling_msg.text = (f"Settling...  {remaining:.1f} s\n\n"
                                  "(cues will begin automatically)")
            settling_msg.draw()
            win.flip()
            core.wait(0.05)
        win.flip()

    # If cueing the sleeping participant, blank the screen.
    if info['cueing_target'] == 'sleep':
        win.color = BG_COLOR
        win.flip()
        op_note = visual.TextStim(win, text='CUEING (sleep target) — blank screen for participant',
                                  color='dimgray', height=18,
                                  pos=(0, -SCREEN_SIZE[1] / 2 + 30))
        op_note.draw()
        win.flip()

    trials_by_id = {t['item_id']: t for t in trials}
    cued_trials  = [trials_by_id[i] for i in cued_ids if i in trials_by_id]

    # Build the queue: rep-major interleaved (rep 0 of all items, then rep 1,...)
    rng = np.random.default_rng(info['seed'] + 1000)
    queue = []
    for rep in range(info['max_cue_repetitions']):
        order = list(range(len(cued_trials)))
        rng.shuffle(order)
        for idx in order:
            queue.append((rep, cued_trials[idx]))

    cue_log          = []
    interruption_log = []
    retired          = set()
    end_state        = 'completed'
    n_play_errors    = 0

    event.clearEvents()    # discard any stale keypresses from setup screens
    resync_eeg()           # resync after operator screens, right before cues start
    q_idx = 0
    while q_idx < len(queue):
        rep, trial = queue[q_idx]
        if trial['item_id'] in retired:
            q_idx += 1
            continue

        # Check for operator keys between cues
        keys = event.getKeys(keyList=[KEY_AROUSAL, KEY_PAUSE,
                                      KEY_QUIT, KEY_JUMP_POSTTEST, KEY_ESC])
        if KEY_ESC in keys:
            raise KeyboardInterrupt("Operator ESC during cueing")
        if keys:
            action = handle_interruption(win, keys[0], info,
                                         interruption_log,
                                         current_item=trial['item_id'])
            if action == 'jump_post':
                end_state = 'jump_post'
                break
            elif action == 'quit':
                end_state = 'quit'
                break
            elif action == 'skip':
                retired.add(trial['item_id'])
                q_idx += 1
                continue
            # else 'resumed' — re-check loop
            continue

        # Play cue — resync before each cue to keep the socket alive across ICIs
        resync_eeg()
        try:
            snd = sound.Sound(trial['sound_file'])
            snd.setVolume(CUEING_VOLUME)
        except Exception as e:
            logging.warning(f"Cue sound load failed for {trial['item_id']}: {e}")
            q_idx += 1
            continue

        t_cue = core.getTime()
        send_trigger(TRIG['cue_onset'], f"cue_{trial['item_id']}_rep{rep}",
                     {'item_id': trial['item_id'], 'rep': rep})

        try:
            snd.play()
        except Exception as e:
            logging.warning(f"Cue sound PLAY failed for {trial['item_id']}: {e}"
                            " — run _audio_setup.py to adapt WAV channel count")
            n_play_errors += 1
            q_idx += 1
            continue

        cue_log.append({
            'cue_index'   : len(cue_log),
            'item_id'     : trial['item_id'],
            'word'        : trial['word'],
            'rep'         : rep,
            'onset_sec'   : t_cue,
            'sound_file'  : trial['sound_file'],
        })

        # Inter-cue interval with live interrupt checking
        ici = float(rng.uniform(*INTER_CUE_INTERVAL_SEC))
        interrupted = wait_with_interrupt_check(ici)

        try:
            snd.stop()
        except Exception:
            pass

        if interrupted is not None:
            action = handle_interruption(win, interrupted, info,
                                         interruption_log,
                                         current_item=trial['item_id'])
            if action == 'jump_post':
                end_state = 'jump_post'
                break
            elif action == 'quit':
                end_state = 'quit'
                break
            elif action == 'skip':
                retired.add(trial['item_id'])

        q_idx += 1

    # Warn loudly if all audio attempts failed (likely a channel-count mismatch)
    if n_play_errors > 0 and len(cue_log) == 0:
        logging.error(
            f"ALL {n_play_errors} cue sound play attempts failed. "
            "Run _audio_setup.py from this paradigm's folder to adapt "
            "WAV files to your audio device's channel count.")

    # Phase bookkeeping
    write_csv(cue_log,          os.path.join(_run_dir, 'cue_log.csv'))
    write_csv(interruption_log, os.path.join(_run_dir, 'interruption_log.csv'))
    send_trigger(TRIG['phase_boundary'], f'cueing_end_{end_state}')
    return cue_log, interruption_log, end_state


def wait_with_interrupt_check(duration_sec: float):
    """
    Sleep for duration_sec, polling for A/P/J/Q every ~10 ms. If one of those
    is pressed, return it immediately. Otherwise return None at the end.
    """
    t_end = core.getTime() + duration_sec
    watch = [KEY_AROUSAL, KEY_PAUSE, KEY_QUIT, KEY_JUMP_POSTTEST, KEY_ESC]
    while core.getTime() < t_end:
        keys = event.getKeys(keyList=watch)
        if KEY_ESC in keys:
            raise KeyboardInterrupt("Operator ESC during cueing wait")
        if keys:
            return keys[0]
        core.wait(0.01)
    return None


def handle_interruption(win, trigger_key, info, interruption_log, current_item):
    """
    Step through: log onset → classify → choose action → (optional settling).
    Returns one of 'resumed', 'skip', 'jump_post', 'quit'.
    """
    t_interrupt = core.getTime()
    trig_name = TRIG['arousal_flag'] if trigger_key == KEY_AROUSAL else TRIG['pause_flag']
    send_trigger(trig_name, f'interrupt_{trigger_key}',
                 {'trigger': trigger_key, 'item_id': current_item})

    # Unblank for the operator — they need to see the prompts
    prev_color = win.color
    win.color = BG_COLOR
    win.flip()

    # --- classify ---
    classify_key = show_text_wait(win,
        "INTERRUPTION — classify the event\n\n"
        "  0  manual / technical pause\n"
        "  1  mild movement\n"
        "  2  eye opening / blink burst\n"
        "  3  vocalization / large movement\n"
        "  4  clear awakening\n"
        "  5  unclear / other",
        wait_keys=('0', '1', '2', '3', '4', '5'))

    classify_map = {'0': 'manual_pause', '1': 'mild_movement',
                    '2': 'eye_open_blink', '3': 'vocalization_large_movement',
                    '4': 'clear_awakening', '5': 'unclear_other'}
    classification = classify_map.get(classify_key, 'unknown')

    # --- action ---
    action_key = show_text_wait(win,
        f"Classification: {classification}\n\n"
        "ACTION:\n"
        f"  R  resume cueing (settling interval = {info['settling_interval']:.1f} s)\n"
        "  K  skip / retire current item\n"
        "  J  jump to post-test\n"
        "  Q  quit",
        wait_keys=(KEY_RESUME, KEY_SKIP_ITEM, KEY_JUMP_POSTTEST, KEY_QUIT))

    action_map = {KEY_RESUME: 'resume', KEY_SKIP_ITEM: 'skip',
                  KEY_JUMP_POSTTEST: 'jump_post', KEY_QUIT: 'quit'}
    action = action_map.get(action_key, 'resume')

    settling_actual = 0.0
    settle_exit = None
    if action == 'resume':
        settling_actual, settle_exit = run_settling_interval(
            win, info['settling_interval'])
        if settle_exit == 'jump_post':
            action = 'jump_post'
        elif settle_exit == 'quit':
            action = 'quit'
        send_trigger(TRIG['resume_from_int'], 'resume_after_settle',
                     {'settle_sec': round(settling_actual, 2)})

    if action == 'jump_post':
        send_trigger(TRIG['jump_to_posttest'], 'jump_requested')

    # Restore background
    win.color = prev_color
    win.flip()

    pause_duration = core.getTime() - t_interrupt

    interruption_log.append({
        'interrupt_time_sec'  : t_interrupt,
        'trigger_key'         : trigger_key,
        'classification'      : classification,
        'action'              : action,
        'settling_requested'  : info['settling_interval'],
        'settling_actual_sec' : settling_actual,
        'settle_exit'         : settle_exit,
        'pause_duration_sec'  : pause_duration,
        'current_item_id'     : current_item,
    })

    # Map action → internal return value
    if action == 'resume':
        return 'resumed'
    return action


def run_settling_interval(win, duration_sec):
    """
    Wait duration_sec with a settling screen. Operator can:
      SPACE — resume early
      J     — jump to post-test
      Q     — quit
    Returns (actual_seconds_waited, exit_reason_or_None).
    """
    if duration_sec <= 0:
        return 0.0, None

    t_start = core.getTime()
    msg = visual.TextStim(win, text='', color=FG_COLOR, height=24,
                          wrapWidth=SCREEN_SIZE[0] * 0.8, alignText='center')
    event.clearEvents()
    watch = [KEY_CONFIRM, KEY_JUMP_POSTTEST, KEY_QUIT, KEY_ESC]

    while True:
        elapsed = core.getTime() - t_start
        remaining = duration_sec - elapsed
        if remaining <= 0:
            return elapsed, None
        msg.text = (f"SETTLING...  {remaining:4.1f} s remaining\n\n"
                    f"SPACE  resume early (participant is stable)\n"
                    f"J      jump to post-test\n"
                    f"Q      quit")
        msg.draw()
        win.flip()

        keys = event.getKeys(keyList=watch)
        if KEY_ESC in keys:
            raise KeyboardInterrupt("Operator ESC during settling")
        if KEY_CONFIRM in keys:
            return core.getTime() - t_start, None
        if KEY_JUMP_POSTTEST in keys:
            return core.getTime() - t_start, 'jump_post'
        if KEY_QUIT in keys:
            return core.getTime() - t_start, 'quit'
        core.wait(0.05)


# ===========================================================================
# SUMMARY
# ===========================================================================

def summarize(pretest, posttest, cue_log, cue_groups):
    """Group-level pre/post stats for end-of-run operator summary."""
    pre_by = {r['item_id']: r for r in pretest}
    post_by = {r['item_id']: r for r in posttest}

    rows = []
    group_order = ['cued_well_learned', 'uncued_well_learned', 'not_well_learned',
                   'fallback_cued', 'fallback_uncued']
    for g in group_order:
        ids = [i for i, gg in cue_groups.items() if gg == g]
        if not ids:
            continue
        pre_errs  = [pre_by[i]['error_px'] for i in ids if i in pre_by]
        post_errs = [post_by[i]['error_px'] for i in ids if i in post_by]
        rows.append({
            'group'              : g,
            'n_items'            : len(ids),
            'n_with_pre'         : len(pre_errs),
            'n_with_post'        : len(post_errs),
            'mean_pre_error_px'  : float(np.mean(pre_errs))  if pre_errs  else np.nan,
            'mean_post_error_px' : float(np.mean(post_errs)) if post_errs else np.nan,
            'improvement_px'     : (float(np.mean(pre_errs)) - float(np.mean(post_errs)))
                                     if pre_errs and post_errs else np.nan,
        })

    # Cueing stats
    cue_counts = {}
    for c in cue_log:
        cue_counts[c['item_id']] = cue_counts.get(c['item_id'], 0) + 1
    cue_stats = {
        'n_cue_presentations': len(cue_log),
        'n_unique_items_cued': len(cue_counts),
        'mean_reps_per_item' : float(np.mean(list(cue_counts.values())))
                                 if cue_counts else 0.0,
    }
    return rows, cue_stats


def show_summary(win, summary_rows, cue_stats, end_state):
    lines = [f"END-OF-RUN SUMMARY  (end_state: {end_state})", ""]
    lines.append(f"Cue presentations delivered : {cue_stats['n_cue_presentations']}")
    lines.append(f"Unique items cued           : {cue_stats['n_unique_items_cued']}")
    lines.append(f"Mean repetitions per item   : {cue_stats['mean_reps_per_item']:.2f}")
    lines.append("")
    lines.append(f"{'group':<22} {'n':>4} {'preE':>7} {'postE':>7} {'Δ':>7}")
    for r in summary_rows:
        pre  = f"{r['mean_pre_error_px']:.1f}"  if not np.isnan(r['mean_pre_error_px'])  else '  —  '
        post = f"{r['mean_post_error_px']:.1f}" if not np.isnan(r['mean_post_error_px']) else '  —  '
        imp  = f"{r['improvement_px']:+.1f}"    if not np.isnan(r['improvement_px'])    else '  —  '
        lines.append(f"{r['group']:<22} {r['n_items']:>4} {pre:>7} {post:>7} {imp:>7}")
    lines.append("")
    lines.append("Lower error = more accurate. Positive Δ = improvement from pre to post.")
    lines.append("")
    lines.append("Press SPACE to exit.")

    show_text_wait(win, "\n".join(lines), height=22)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    global _session_info, _run_dir

    # Basic console logging
    py_logging.basicConfig(level=py_logging.INFO,
                           format='%(asctime)s %(levelname)s %(message)s')

    info = run_setup_dialog()
    _session_info = info
    globals()['EEG_MODE'] = bool(info['eeg_mode'])

    # Set up output dir and per-run psychopy log
    _run_dir = ensure_run_dir(info['subject_number'], info['session'])
    logging.LogFile(os.path.join(_run_dir, 'psychopy.log'),
                    level=logging.INFO, filemode='w')

    # Persist the resolved session info
    with open(os.path.join(_run_dir, 'session_info.json'), 'w') as f:
        json.dump(info, f, indent=2, default=str)

    trials = get_or_create_trials(info)

    connect_netstation_if_enabled()

    win = visual.Window(size=SCREEN_SIZE, units=SCREEN_UNITS,
                        color=BG_COLOR, fullscr=bool(info['fullscreen']),
                        allowGUI=True)

    # Prebuild grid lines ONCE (line stims are cheap to draw, expensive to
    # construct every frame). Passed into each phase; each phase decides
    # whether to draw them based on SHOW_GRID_IN_{STUDY,TEST}.
    grid_stims = build_grid_stims(win)

    pretest = []
    posttest = []
    cue_log = []
    cue_groups = {}
    end_state = 'completed'

    try:
        # Operator briefing
        show_text_wait(win, EXPERIMENTER_BRIEFING)

        sp = info['start_phase']
        phases_to_run = []
        for p in ('study', 'pre_test', 'cueing', 'post_test'):
            phases_to_run.append(p)
        # Skip up to the requested start phase
        start_idx = phases_to_run.index(sp)
        phases_to_run = phases_to_run[start_idx:]

        if 'study' in phases_to_run:
            run_study_phase(win, trials, info, grid_stims=grid_stims)

        if 'pre_test' in phases_to_run:
            pretest = run_memory_test(win, trials, info, 'pre_test',
                                      grid_stims=grid_stims)

        if 'cueing' in phases_to_run:
            cue_groups = assign_cue_groups(trials, pretest, info)
            # Persist group assignment early
            write_csv([{'item_id': k, 'group': v} for k, v in cue_groups.items()],
                      os.path.join(_run_dir, 'cue_groups.csv'))
            cue_log, _, end_state = run_cueing_phase(win, trials, info, cue_groups)

        if 'post_test' in phases_to_run and end_state != 'quit':
            posttest = run_memory_test(win, trials, info, 'post_test',
                                       grid_stims=grid_stims)

        summary_rows, cue_stats = summarize(pretest, posttest, cue_log, cue_groups)
        write_csv(summary_rows, os.path.join(_run_dir, 'summary_by_group.csv'))
        with open(os.path.join(_run_dir, 'summary_cueing.json'), 'w') as f:
            json.dump(cue_stats, f, indent=2)

        show_summary(win, summary_rows, cue_stats, end_state)

    except KeyboardInterrupt as e:
        logging.warning(f'Aborted: {e}')
    finally:
        # Always persist the event log and shut things down cleanly
        write_csv(_event_log, os.path.join(_run_dir, 'event_log.csv'))
        disconnect_netstation()
        try:
            win.close()
        except Exception:
            pass
        core.quit()


if __name__ == '__main__':
    main()
