"""Version line shown in the sidebar of every app.

Usage at the top of an app (after st.set_page_config):

    from version_info import show_version
    show_version(__file__)

The version is read from git, so it updates with every commit.
"""
import datetime
import os
import subprocess

import streamlit as st


@st.cache_data(ttl=30)  # refresh every 30 s, so slider moves don't call git each time
def get_version_info(app_path):
    app_dir = os.path.dirname(os.path.abspath(app_path))

    def git(*args):
        return subprocess.run(["git", "-C", app_dir, *args], capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip()
    try:
        commit, date = git("log", "-1", "--format=%h|%cd", "--date=format:%Y-%m-%d %H:%M",
                           "--", os.path.abspath(app_path)).split("|")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        changed = git("status", "--porcelain", "--", os.path.abspath(app_path))
        text = f"Version {commit} · {date} · branch {branch}"
        if changed:
            text += " · ✏️ with local changes, not yet committed"
        return text
    except Exception:
        # No git available: fall back to the file's last modification time
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(app_path))
        return f"File last modified {mtime:%Y-%m-%d %H:%M}"


def show_version(app_path):
    st.sidebar.caption(get_version_info(app_path))
