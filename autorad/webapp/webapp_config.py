import os
from pathlib import Path

WEBAPP_DIR = Path(os.path.dirname(__file__))
TEMPLATE_DIR = WEBAPP_DIR / "templates"
IS_DEMO = False
if "INPUT_DIR" not in os.environ:
    IS_DEMO = True
