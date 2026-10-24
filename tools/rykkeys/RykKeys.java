// RykKeys: the headset-button bridge for `ryk --serve` (docs/headset-control.md).
//
//   app_process / RykKeys SOCKET
//
// Owns a MediaSession as the Termux uid, so the headset's media keys arrive here
// even with the screen off. Each key becomes a control verb on a long-lived
// `watch` connection to the daemon's socket; the daemon's `state` lines are
// mirrored into the session's PlaybackState. Exits when the daemon closes the
// connection (end of the rewind window, daemon gone), so nothing has to stop it.
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
import android.os.Looper;
import android.view.KeyEvent;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.lang.reflect.Constructor;

public class RykKeys {
    static final String PKG = "com.termux";

    static OutputStream out;
    static MediaSession session;

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

    /**
     * The verb for a key, or null for keys we consume and ignore.
     *
     * The Sony XM4 resolves its taps itself: double tap arrives as PAUSE or PLAY
     * (it alternates by its own state), swipes as NEXT / PREVIOUS. A wired
     * headset sends HEADSETHOOK / PLAY_PAUSE; counting its clicks comes later.
     */
    static String verbFor(int keyCode) {
        switch (keyCode) {
            case KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE:
            case KeyEvent.KEYCODE_MEDIA_PLAY:
            case KeyEvent.KEYCODE_MEDIA_PAUSE:
            case KeyEvent.KEYCODE_HEADSETHOOK:
                return "toggle";
            case KeyEvent.KEYCODE_MEDIA_NEXT:
            case KeyEvent.KEYCODE_MEDIA_PREVIOUS:
                return "back";
            case KeyEvent.KEYCODE_MEDIA_STOP:
                return "cancel";
            default:
                return null;
        }
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
                    // One action per press: the key-down, not its repeats or the key-up.
                    if (e.getAction() != KeyEvent.ACTION_DOWN || e.getRepeatCount() > 0) return true;
                    String verb = verbFor(e.getKeyCode());
                    log("key " + KeyEvent.keyCodeToString(e.getKeyCode())
                            + (verb == null ? " (ignored)" : " = " + verb));
                    if (verb != null) send(verb);
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

    static int run(String[] a) throws Exception {
        if (a.length < 1) {
            log("usage: RykKeys SOCKET");
            return 1;
        }
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
