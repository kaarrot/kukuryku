// Throwaway spike: can an app_process jar running as the Termux uid own a
// MediaSession that receives media keys while pacat is playing?
//
//   MediaSpike session [SECS] [guided]    ...with `guided`, prompt for one gesture at a time
//                                         and print which key each one produced
//   MediaSpike session [SECS] [handoff]   create + activate a session, log media keys;
//                                         with `handoff`, on the first key release the
//                                         session and re-dispatch that key to whoever
//                                         held the buttons before
//   MediaSpike key NAME [N] [GAP_MS]   dispatch a media key like any app can, N times (default 1)
//                               GAP_MS apart (default 100): STOP PLAY_PAUSE PLAY PAUSE HOOK NEXT
//                               PREVIOUS REWIND FAST_FORWARD
//
// Same hidden-API-by-reflection style as TdAlarm.java.

import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.media.session.MediaSession;
import android.media.session.MediaSessionManager;
import android.media.session.PlaybackState;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.SystemClock;
import android.view.KeyEvent;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

public class MediaSpike {
    static final String PKG = "com.termux";

    public static void main(String[] a) throws Exception {
        try {
            main2(a);
        } catch (Throwable t) {
            Throwable c = t;
            while (c.getCause() != null) c = c.getCause();
            System.out.println("FAILED: " + t);
            System.out.println("ROOT: " + c);
            for (StackTraceElement e : c.getStackTrace()) System.out.println("  at " + e);
            System.out.flush();
            System.exit(1);
        }
    }

    static void main2(String[] a) throws Exception {
        if (a.length >= 1 && a[0].equals("key")) {
            key(a.length > 1 ? a[1] : "STOP", a.length > 2 ? Integer.parseInt(a[2]) : 1,
                    a.length > 3 ? Integer.parseInt(a[3]) : 100);
        } else {
            session(a.length > 1 ? Integer.parseInt(a[1]) : 30,
                    a.length > 2 && a[2].equals("handoff"),
                    a.length > 2 && a[2].equals("guided"));
        }
    }

    static void log(String s) {
        System.out.println(System.currentTimeMillis() % 100000 + " " + s);
        System.out.flush();
    }

    // A context that says it is com.termux. MediaSession reads the package
    // name from the context it is given, and system_server checks it against
    // the calling uid.
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

    static void session(int secs, final boolean handoff, final boolean guided) throws Exception {
        Looper.prepareMainLooper();
        Class<?> at = Class.forName("android.app.ActivityThread");
        at.getMethod("initializeMainlineModules").invoke(null);
        Object thread = at.getMethod("systemMain").invoke(null);
        Context sys = (Context) at.getMethod("getSystemContext").invoke(thread);
        log("system context ok, pkg=" + sys.getPackageName());

        final MediaSession s = new MediaSession(new Fake(sys), "ryk-spike");
        s.setFlags(MediaSession.FLAG_HANDLES_MEDIA_BUTTONS
                | MediaSession.FLAG_HANDLES_TRANSPORT_CONTROLS);
        s.setCallback(new MediaSession.Callback() {
            @Override
            public boolean onMediaButtonEvent(Intent i) {
                KeyEvent e = (KeyEvent) i.getParcelableExtra(Intent.EXTRA_KEY_EVENT);
                log("onMediaButtonEvent " + e);
                if (guided && e != null && e.getAction() == KeyEvent.ACTION_DOWN) {
                    cur.append(' ').append(KeyEvent.keyCodeToString(e.getKeyCode())
                            .replace("KEYCODE_", ""));
                }
                if (handoff && e != null && e.getAction() == KeyEvent.ACTION_DOWN) {
                    log("handoff: releasing, then re-dispatching keyCode " + e.getKeyCode());
                    s.release();
                    try {
                        dispatch(e.getKeyCode());
                    } catch (Exception x) {
                        log("dispatch failed: " + x);
                    }
                    System.exit(0);
                }
                return true;
            }
        });
        s.setPlaybackState(new PlaybackState.Builder()
                .setActions(PlaybackState.ACTION_PLAY_PAUSE | PlaybackState.ACTION_PLAY
                        | PlaybackState.ACTION_PAUSE | PlaybackState.ACTION_SKIP_TO_NEXT
                        | PlaybackState.ACTION_SKIP_TO_PREVIOUS | PlaybackState.ACTION_STOP)
                .setState(PlaybackState.STATE_PLAYING, 0, 1f)
                .build());
        s.setActive(true);
        log("session created, active=" + s.isActive() + " token=" + s.getSessionToken());
        if (guided) {
            final Handler gh = new Handler(Looper.getMainLooper());
            log("guided test: starts in 3 s, " + STEPS.length + " gestures, "
                    + STEP_MS / 1000.0 + " s each");
            gh.postDelayed(new Runnable() {
                public void run() {
                    runSteps(gh, 0, s);
                }
            }, 3000);
            Looper.loop();
            return;
        }
        new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
            public void run() {
                log("releasing");
                s.release();
                System.exit(0);
            }
        }, secs * 1000L);
        Looper.loop();
    }

    static void key(String name, int times, int gapMs) throws Exception {
        int code;
        switch (name) {
            case "PLAY_PAUSE": code = 85; break;
            case "NEXT": code = 87; break;
            case "PREVIOUS": code = 88; break;
            case "PLAY": code = 126; break;
            case "PAUSE": code = 127; break;
            case "HOOK": code = 79; break;
            case "REWIND": code = 89; break;
            case "FAST_FORWARD": code = 90; break;
            default: code = 86; // STOP
        }
        Object sm = service("media_session", "android.media.session.ISessionManager");
        for (Method m : sm.getClass().getMethods()) {
            if (m.getName().equals("dispatchMediaKeyEvent")) {
                log("method " + m);
            }
        }
        for (int n = 0; n < times; n++) {
            if (n > 0) Thread.sleep(gapMs);
            long now = SystemClock.uptimeMillis();
            for (int action = 0; action <= 1; action++) {
                KeyEvent e = new KeyEvent(now, now, action, code, 0);
                call(sm, "dispatchMediaKeyEvent", PKG, false, e, false);
            }
        }
        log("dispatched " + name + " x" + times);
    }

    static final String[] STEPS = {
        "DOUBLE TAP on the right ear cup",
        "DOUBLE TAP again",
        "TRIPLE TAP",
        "SINGLE TAP",
        "SWIPE FORWARD",
        "SWIPE BACK",
        "SWIPE FORWARD and HOLD (about 2 s)",
        "SWIPE BACK and HOLD (about 2 s)",
    };
    static final int STEP_MS = System.getenv("STEP_MS") != null
            ? Integer.parseInt(System.getenv("STEP_MS")) : 7000;
    static final StringBuilder cur = new StringBuilder();
    static final StringBuilder all = new StringBuilder();

    static void runSteps(final Handler h, final int i, final MediaSession s) {
        if (i > 0) {
            String r = cur.length() == 0 ? "(nothing)" : cur.toString().trim();
            String line = STEPS[i - 1] + " -> " + r;
            log("RESULT " + line);
            all.append(line).append("\n");
        }
        if (i == STEPS.length) {
            log("SUMMARY\n" + all);
            s.release();
            System.exit(0);
        }
        cur.setLength(0);
        log("=== STEP " + (i + 1) + "/" + STEPS.length + ": " + STEPS[i]
                + " - do it now, once ===");
        h.postDelayed(new Runnable() {
            public void run() {
                runSteps(h, i + 1, s);
            }
        }, STEP_MS);
    }

    static void dispatch(int code) throws Exception {
        Object sm = service("media_session", "android.media.session.ISessionManager");
        long now = SystemClock.uptimeMillis();
        for (int action = 0; action <= 1; action++) {
            KeyEvent e = new KeyEvent(now, now, action, code, 0);
            call(sm, "dispatchMediaKeyEvent", PKG, false, e, false);
        }
    }

    static Object service(String name, String iface) throws Exception {
        IBinder b = (IBinder) Class.forName("android.os.ServiceManager")
                .getMethod("getService", String.class).invoke(null, name);
        return Class.forName(iface + "$Stub")
                .getMethod("asInterface", IBinder.class).invoke(null, b);
    }

    static Object call(Object target, String name, Object... args) throws Exception {
        for (Method m : target.getClass().getMethods()) {
            if (m.getName().equals(name) && m.getParameterTypes().length == args.length) {
                m.setAccessible(true);
                return m.invoke(target, args);
            }
        }
        throw new NoSuchMethodException(name + " with " + args.length + " arguments");
    }
}
