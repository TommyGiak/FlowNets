import os
import site

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
site.addsitedir(ROOT)
