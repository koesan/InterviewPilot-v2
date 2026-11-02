#!/bin/sh
# Global screenshot key for the interview assistant (left Alt tapped alone, a release bind in Hyprland).
# First press arms, second press captures the screen you are looking at.
PIDFILE="${XDG_RUNTIME_DIR:-/tmp}/interview-assistant.pid"
[ -f "$PIDFILE" ] && kill -USR2 "$(cat "$PIDFILE")"
