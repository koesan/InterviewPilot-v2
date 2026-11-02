#!/bin/sh
# Global LISTENING/PAUSED toggle (bind to a key in Hyprland).
PIDFILE="${XDG_RUNTIME_DIR:-/tmp}/interview-assistant.pid"
[ -f "$PIDFILE" ] && kill -USR1 "$(cat "$PIDFILE")"
