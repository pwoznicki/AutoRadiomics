import json
import os
import sys

import streamlit.bootstrap
import streamlit.config


def _patch_streamlit_print_url():
    _original_print_url = streamlit.bootstrap._print_url

    def _new_print_url(is_running_hello: bool) -> None:
        port = int(streamlit.config.get_option("browser.serverPort"))

        sys.stdout.flush()
        print(json.dumps({"port": port}))
        sys.stdout.flush()

        _original_print_url(is_running_hello)

    streamlit.bootstrap._print_url = _new_print_url


if __name__ == "__main__":
    fname = os.path.join(os.path.dirname(__file__), "webapp/app.py")
    _patch_streamlit_print_url()

    conf = {
        "server.fileWatcherType": "none",
        "server.headless": True,
    }
    streamlit.bootstrap.load_config_options(flag_options=conf)

    streamlit.bootstrap.run(fname, "", args=[], flag_options={})
