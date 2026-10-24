// RykKeys: the headset-button bridge for `ryk --serve` (docs/headset-control.md).
//
//   app_process / RykKeys SOCKET     run: bridge the headset buttons to the daemon
//   app_process / RykKeys check      diagnose: do the hidden-API steps still work? (exit 0/1)
//
// Owns a MediaSession as the Termux uid, so the headset's media keys arrive here
// even with the screen off. Each key becomes a control verb on a long-lived
// `watch` connection to the daemon's socket; the daemon's `state` lines are
// mirrored into the session's PlaybackState. Exits when the daemon closes the
// connection (end of the rewind window, daemon gone), so nothing has to stop it.
//
// Settings (environment):
//   RYK_KEYS_MAP       gesture:verb pairs, e.g. "tap:pause,stop:none". Gestures: tap
//                      (double tap on the XM4, one click on a wired headset), skip
//                      (NEXT/PREVIOUS, or two or more clicks), stop (STOP). Verbs: pause
//                      resume toggle back cancel rewind none. Defaults: tap:toggle
//                      skip:back stop:cancel.
//   RYK_KEYS_CLICK_MS  how long to wait for more clicks of a wired headset (350)
//
// Exit codes: 0 = the daemon went away (normal); 1 = something failed here
// (hidden API, session). An uncaught exception in a process like this is a
// silent SIGKILL, so everything is wrapped and printed.
//
// Same hidden-API-by-reflection style and build recipe as TdAlarm.java
// (tools/rykkeys/build.sh: ecj + dx).

import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.media.session.MediaSession;
import android.media.session.MediaSessionManager;
import android.media.session.PlaybackState;
import android.net.LocalSocket;
import android.net.LocalSocketAddress;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.view.KeyEvent;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.lang.reflect.Constructor;
import java.util.HashMap;

public class RykKeys {
    static final String PKG = "com.termux";

    static OutputStream out;
    static MediaSession session;
    static Handler handler;

    /** gesture -> verb */
    static final HashMap<String, String> MAP = new HashMap<String, String>();
    static long clickMs = 350;
    /** If a click's key-up never comes, count it anyway after this long. */
    static final long CLICK_FALLBACK_MS = 1200;
    static int clicks;

    public static void main(String[] a) {
        int code = 1;
        try {
            code = run(a);
        } catch (Throwable t) {
            fail(t);
        }
        System.out.flush();
        System.exit(code);
    }

    static void log(String s) {
        System.out.println(System.currentTimeMillis() % 100000 + " " + s);
        System.out.flush();
    }

    static void fail(Throwable t) {
        Throwable c = t;
        while (c.getCause() != null) c = c.getCause();
        log("FAILED: " + t);
        log("ROOT: " + c);
        for (StackTraceElement e : c.getStackTrace()) log("  at " + e);
    }

    // ---- what a key means ----------------------------------------------------

    static final String[] GESTURES = {"tap", "skip", "stop"};
    static final String[] VERBS = {"pause", "resume", "toggle", "back", "cancel", "rewind", "none"};

    static boolean in(String[] set, String v) {
        for (String x : set) if (x.equals(v)) return true;
        return false;
    }

    static void loadSettings() {
        MAP.put("tap", "toggle");
        MAP.put("skip", "back");
        MAP.put("stop", "cancel");
        String spec = System.getenv("RYK_KEYS_MAP");
        if (spec != null) {
            for (String pair : spec.split(",")) {
                pair = pair.trim();
                if (pair.length() == 0) continue;
                String[] kv = pair.split(":");
                if (kv.length == 2 && in(GESTURES, kv[0].trim()) && in(VERBS, kv[1].trim())) {
                    MAP.put(kv[0].trim(), kv[1].trim());
                } else {
                    log("ignoring RYK_KEYS_MAP entry '" + pair + "'");
                }
            }
        }
        String ms = System.getenv("RYK_KEYS_CLICK_MS");
        if (ms != null) {
            try {
                long v = Long.parseLong(ms.trim());
                if (v >= 50 && v <= 2000) clickMs = v;
                else log("ignoring RYK_KEYS_CLICK_MS=" + ms + " (50..2000)");
            } catch (NumberFormatException e) {
                log("ignoring RYK_KEYS_CLICK_MS=" + ms);
            }
        }
        log("map tap=" + MAP.get("tap") + " skip=" + MAP.get("skip") + " stop=" + MAP.get("stop")
                + " click=" + clickMs + "ms");
    }

    /**
     * A key that arrives already resolved, as a gesture (null: consumed, ignored).
     *
     * The Sony XM4 resolves its taps itself: double tap arrives as PAUSE or PLAY
     * (it alternates by its own state), swipes as NEXT / PREVIOUS.
     */
    static String gestureFor(int keyCode) {
        switch (keyCode) {
            case KeyEvent.KEYCODE_MEDIA_PLAY:
            case KeyEvent.KEYCODE_MEDIA_PAUSE:
                return "tap";
            case KeyEvent.KEYCODE_MEDIA_NEXT:
            case KeyEvent.KEYCODE_MEDIA_PREVIOUS:
                return "skip";
            case KeyEvent.KEYCODE_MEDIA_STOP:
                return "stop";
            default:
                return null;
        }
    }

    /**
     * Keys of a wired headset's one button arrive as raw clicks: PLAY_PAUSE or
     * HEADSETHOOK, a down and an up each. They are counted: one click is a tap,
     * two or more are a skip. (Android does not turn double clicks into NEXT.)
     */
    static boolean isClickKey(int keyCode) {
        return keyCode == KeyEvent.KEYCODE_HEADSETHOOK
                || keyCode == KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE;
    }

    static final Runnable FIRE = new Runnable() {
        public void run() {
            try {
                int n = clicks;
                clicks = 0;
                if (n > 0) {
                    log(n + " click" + (n == 1 ? "" : "s"));
                    perform(n == 1 ? "tap" : "skip");
                }
            } catch (Throwable t) {
                fail(t);
            }
        }
    };

    static void arm(long ms) {
        handler.removeCallbacks(FIRE);
        handler.postDelayed(FIRE, ms);
    }

    static void onClickKey(KeyEvent e) {
        if (e.getAction() == KeyEvent.ACTION_DOWN) {
            if (e.getRepeatCount() == 0) {
                clicks++;
                arm(CLICK_FALLBACK_MS);
            }
        } else if (e.getAction() == KeyEvent.ACTION_UP && clicks > 0) {
            arm(clickMs);
        }
    }

    static void perform(String gesture) {
        String verb = MAP.get(gesture);
        if (verb == null || verb.equals("none")) {
            log(gesture + " is unmapped");
            return;
        }
        send(verb);
    }

    static void send(String verb) {
        try {
            synchronized (RykKeys.class) {
                out.write((verb + "\n").getBytes("UTF-8"));
                out.flush();
            }
            log("-> " + verb);
        } catch (IOException e) {
            log("daemon connection lost while sending " + verb + ": " + e);
            quit(0);
        }
    }

    static void quit(int code) {
        try {
            if (session != null) session.release();
        } catch (Throwable ignored) {
        }
        System.out.flush();
        System.exit(code);
    }

    // ---- the session ---------------------------------------------------------

    // A context that says it is com.termux. MediaSession reads the package name
    // from the context it is given, and system_server checks it against the
    // calling uid. (Proved in tools/headset-spike.)
    static class Fake extends ContextWrapper {
        Fake(Context base) {
            super(base);
        }

        @Override
        public String getPackageName() {
            return PKG;
        }

        public String getOpPackageName() {
            return PKG;
        }

        @Override
        public Context getApplicationContext() {
            return this;
        }

        @Override
        public Object getSystemService(String name) {
            if (Context.MEDIA_SESSION_SERVICE.equals(name)) {
                try {
                    Constructor<MediaSessionManager> c =
                            MediaSessionManager.class.getDeclaredConstructor(Context.class);
                    c.setAccessible(true);
                    return c.newInstance(this);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            return super.getSystemService(name);
        }
    }

    static final long ACTIONS = PlaybackState.ACTION_PLAY_PAUSE | PlaybackState.ACTION_PLAY
            | PlaybackState.ACTION_PAUSE | PlaybackState.ACTION_SKIP_TO_NEXT
            | PlaybackState.ACTION_SKIP_TO_PREVIOUS | PlaybackState.ACTION_STOP;

    /** Mirror the daemon's state: only "playing" is a playing session. */
    static void mirror(String state) {
        boolean playing = state.equals("playing");
        session.setPlaybackState(new PlaybackState.Builder()
                .setActions(ACTIONS)
                .setState(playing ? PlaybackState.STATE_PLAYING : PlaybackState.STATE_PAUSED,
                        PlaybackState.PLAYBACK_POSITION_UNKNOWN, playing ? 1f : 0f)
                .build());
    }

    static void createSession() throws Exception {
        Looper.prepareMainLooper();
        handler = new Handler(Looper.getMainLooper());
        Class<?> at = Class.forName("android.app.ActivityThread");
        at.getMethod("initializeMainlineModules").invoke(null);
        Object thread = at.getMethod("systemMain").invoke(null);
        Context sys = (Context) at.getMethod("getSystemContext").invoke(thread);

        session = new MediaSession(new Fake(sys), "ryk");
        session.setFlags(MediaSession.FLAG_HANDLES_MEDIA_BUTTONS
                | MediaSession.FLAG_HANDLES_TRANSPORT_CONTROLS);
        session.setCallback(new MediaSession.Callback() {
            @Override
            public boolean onMediaButtonEvent(Intent i) {
                try {
                    KeyEvent e = (KeyEvent) i.getParcelableExtra(Intent.EXTRA_KEY_EVENT);
                    if (e == null) return true;
                    if (isClickKey(e.getKeyCode())) {
                        onClickKey(e);
                        return true;
                    }
                    // One action per press: the key-down, not its repeats or the key-up.
                    if (e.getAction() != KeyEvent.ACTION_DOWN || e.getRepeatCount() > 0) return true;
                    String gesture = gestureFor(e.getKeyCode());
                    log("key " + KeyEvent.keyCodeToString(e.getKeyCode())
                            + (gesture == null ? " (ignored)" : " = " + gesture));
                    if (gesture != null) perform(gesture);
                } catch (Throwable t) {
                    fail(t);
                }
                return true; // every media key is ours while the helper lives
            }
        });
        mirror("playing");
        session.setActive(true);
        log("session active");
    }

    // ---- the daemon connection -----------------------------------------------

    /**
     * `RykKeys check`: the same hidden-API steps the helper needs, one line each,
     * so an Android update that breaks one is a one-command diagnosis.
     */
    static int check() {
        log("android " + Build.VERSION.RELEASE + " (sdk " + Build.VERSION.SDK_INT + "), "
                + Build.MANUFACTURER + " " + Build.MODEL);
        String step = "start";
        try {
            step = "Looper.prepareMainLooper";
            Looper.prepareMainLooper();
            Class<?> at = Class.forName("android.app.ActivityThread");
            step = "ActivityThread.initializeMainlineModules";
            at.getMethod("initializeMainlineModules").invoke(null);
            log("ok   " + step);
            step = "ActivityThread.systemMain";
            Object thread = at.getMethod("systemMain").invoke(null);
            log("ok   " + step);
            step = "ActivityThread.getSystemContext";
            Context sys = (Context) at.getMethod("getSystemContext").invoke(thread);
            log("ok   " + step);
            step = "MediaSessionManager(context)";
            Context fake = new Fake(sys);
            fake.getSystemService(Context.MEDIA_SESSION_SERVICE);
            log("ok   " + step);
            step = "MediaSession as " + PKG;
            MediaSession s = new MediaSession(fake, "ryk-check");
            s.setFlags(MediaSession.FLAG_HANDLES_MEDIA_BUTTONS
                    | MediaSession.FLAG_HANDLES_TRANSPORT_CONTROLS);
            s.setActive(true);
            boolean active = s.isActive();
            s.release();
            if (!active) throw new IllegalStateException("the session did not become active");
            log("ok   " + step);
            log("headset check: OK");
            return 0;
        } catch (Throwable t) {
            log("FAIL " + step);
            fail(t);
            log("headset check: FAILED at " + step);
            return 1;
        }
    }

    static int run(String[] a) throws Exception {
        if (a.length < 1) {
            log("usage: RykKeys SOCKET | check");
            return 1;
        }
        if (a[0].equals("check")) return check();
        loadSettings();
        LocalSocket sock = new LocalSocket();
        try {
            sock.connect(new LocalSocketAddress(a[0], LocalSocketAddress.Namespace.FILESYSTEM));
        } catch (IOException e) {
            log("no daemon at " + a[0] + ": " + e); // it went away before we started
            return 0;
        }
        out = sock.getOutputStream();
        final BufferedReader in = new BufferedReader(new InputStreamReader(sock.getInputStream(), "UTF-8"));
        out.write("watch\n".getBytes("UTF-8"));
        out.flush();
        String ok = in.readLine();
        if (!"ok".equals(ok)) {
            log("watch refused: " + ok);
            return ok == null ? 0 : 1;
        }

        createSession();

        Thread reader = new Thread(new Runnable() {
            public void run() {
                try {
                    String line;
                    while ((line = in.readLine()) != null) {
                        if (line.startsWith("state ")) {
                            log("<- " + line);
                            mirror(line.substring(6).trim());
                        } else {
                            log("<- " + line); // the answer to a verb
                        }
                    }
                    log("daemon closed the connection");
                } catch (Throwable t) {
                    log("daemon connection lost: " + t);
                }
                quit(0);
            }
        }, "ryk-keys-reader");
        reader.setDaemon(true);
        reader.start();

        Looper.loop();
        return 0;
    }
}
