import re
import string
import subprocess
from collections import Counter, deque
import colorlog

from fasta import FastA
import numpy as np
import pandas as pd
import pylab
logger = colorlog.getLogger(__name__)

__all__ = ["DNA", "RNA", "Repeats", "Sequence"]
